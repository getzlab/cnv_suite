import os
from pathlib import Path
from typing import Union
from .cnv_helper_methods import (
    get_segment_interval_trees,
    calc_absolute_cn,
    calc_cn_levels,
    calc_avg_cn,
    return_seg_data_at_loci,
    apply_segment_data_to_df,
)
from .simulation_utils import switch_contigs

PathLike = Union[os.PathLike, str]
