from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


import simulate
from simulate.cnv_profile import CNV_Profile, simulate_coverage_and_depth
from utils import PathLike
from utils.simulation_utils import dump_tsv


BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "cnv_data"
OUT_PATH = BASE_PATH / "out"
CYTOBAND_PATH = DATA_PATH / "cytoBand.hg38.txt"
CHROMOSOME_MAP = dict(zip(["chr" + str(x) for x in list(range(1, 23)) + ["X", "Y"]], range(1, 25)))


def parse_cytoband(cytoband):
    # some cytoband files have a header, some don't; we need to check
    has_header = False
    with open(cytoband, "r") as f:
        if f.readline().startswith("chr\t"):
            has_header = True

    cband = pd.read_csv(cytoband, sep="\t", names=["chr", "start", "end", "band", "stain"] if not has_header else None)
    cband["chr"] = cband["chr"].apply(lambda x: CHROMOSOME_MAP[x])
    chrs = cband["chr"].unique()
    ints = dict(zip(chrs, [{0} for _ in range(0, len(chrs))]))
    last_end = None
    last_stain = None
    last_chrom = None
    for _, chrom, start, end, _, stain in cband.itertuples():
        if start == 0:
            if last_end is not None:
                ints[last_chrom].add(last_end)
        if stain == "acen" and last_stain != "acen":
            ints[chrom].add(start)
        if stain != "acen" and last_stain == "acen":
            ints[chrom].add(start)

        last_end = end
        last_stain = stain
        last_chrom = chrom
    ints[chrom].add(end)

    CI = np.full([len(ints), 4], 0)
    for c in chrs:
        CI[c - 1, :] = sorted(ints[c])

    return pd.DataFrame(
        np.c_[np.tile(np.c_[np.r_[1:25]], [1, 2]).reshape(-1, 1), CI.reshape(-1, 2)], columns=["chr", "start", "end"]
    )


def make_read_bed(out_path: PathLike, snv_vcf: PathLike, mean_reads: float):
    """
    Makes a BED file containing a random number of reads for each SNV.
    The reads are samples according to a Poisson distribution with the given mean.
    """
    snv_df = pd.read_csv(
        snv_vcf,
        sep="\t",
        comment="#",
        header=None,
        names=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "test"],
    )

    chrom = []
    pos = []

    for _, row in tqdm(snv_df.iterrows(), total=snv_df.shape[0]):
        chrom.append(row[0])
        pos.append(row[1])

    reads = np.random.poisson(mean_reads, size=snv_df.shape[0])

    res = pd.DataFrame({"CHROM": chrom, "POS": pos, "READS": reads})

    res.to_csv(out_path, sep="\t", index=False)


def simulate_genome(num_subclones=0, **kwargs) -> CNV_Profile:
    # cband = pd.read_csv(CYTOBAND_PATH, sep="\t", names=["chr", "start", "end", "band", "stain"])
    centromere_df = parse_cytoband(CYTOBAND_PATH)
    centromere_df.loc[centromere_df[centromere_df["start"] == 0].index, "arm"] = "p"
    centromere_df.loc[centromere_df[centromere_df["start"] != 0].index, "arm"] = "q"

    centromere_spec_df = centromere_df.set_index(["chr", "arm"]).unstack()[[("start", "q"), ("end", "p")]]  # type: ignore
    centromere_spec_df["avg"] = centromere_spec_df.mean(axis=1).astype(int)
    centromere_spec_df["list"] = centromere_spec_df.apply(
        lambda x: [int(x[("end", "p")]), int(x[("start", "q")])], axis=1
    )
    centromere_spec_df.columns = centromere_spec_df.columns.droplevel(1)
    centromere_spec_df.index = centromere_spec_df.index.map(str)
    centromere_avg_center = centromere_spec_df.to_dict()["avg"]
    # centromere_span_center = centromere_spec_df.to_dict()["list"]

    default_profile = CNV_Profile(num_subclones, csize=DATA_PATH / "NA12878_csizes.tsv", cent_loc=centromere_avg_center)

    default_profile.add_cnv_events(**kwargs)
    default_profile._calculate_cnv_profile()
    default_profile._calculate_df_profiles()

    return default_profile


def make_read_depth(vcf: PathLike, read_depth_lambda: float) -> pd.DataFrame:
    snv_df = pd.read_csv(
        vcf,
        sep="\t",
        comment="#",
        names=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "NA12878"],
    )
    snv_df = snv_df.query('NA12878 in ("1|0", "0|1")')[["CHROM", "POS"]]
    snv_df["DEPTH"] = snv_df.apply(lambda _: np.random.poisson(read_depth_lambda), axis=1)  # type: ignore
    return snv_df


if __name__ == "__main__":
    # dump_tsv(make_read_depth(DATA_PATH / "NA12878.vcf", 50), OUT_PATH / "read_depth.tsv")

    simulate.chromosome.main()

    # name = "sim"
    # purity = 0.7

    # prof = simulate_genome(
    #     num_subclones=3, arm_num=20, focal_num=600, p_whole=0.6, ratio_clonal=0.5, median_focal_length=1.8 * 10**6
    # )

    # prof.to_pickle(OUT_PATH / f"{name}_profile.pickle")

    # dump_tsv(prof.generate_random_mutation_file(1e-6, 25), OUT_PATH / f"{name}_mutation_plan.tsv")
    # dump_tsv(
    #     prof.generate_mutations(OUT_PATH / f"{name}_mutation_plan.tsv", purity), OUT_PATH / f"{name}_mutations.tsv"
    # )
    # (OUT_PATH / f"{name}_mutation_plan.tsv").unlink()

    # simulate_coverage_and_depth(
    #     (OUT_PATH / f"{name}_profile.pickle").open("rb"),
    #     DATA_PATH / "NA12878_platinum_realigned_covcollect.bed",
    #     DATA_PATH / "NA12878.vcf",
    #     OUT_PATH / "read_depth.tsv",
    #     purity,
    #     OUT_PATH / f"{name}_coverage.vcf",
    #     OUT_PATH / f"{name}_output_hets.vcf",
    #     do_parallel=False,
    # )

    # print("Finished!")
