import os
from typing import Union

PathLike = Union[os.PathLike, str]


from .cnv_helper_methods import (
    get_segment_interval_trees,
    calc_absolute_cn,
    calc_cn_levels,
    calc_avg_cn,
    return_seg_data_at_loci,
    apply_segment_data_to_df,
)


from .simulation_utils import switch_contigs

_UNIQUE_ID = 0


def gen_id() -> int:
    """
    Generates a unique ID for an object that is guaranteed to be different from all other unique IDs.
    IDs are created in ascending order.
    """
    global _UNIQUE_ID
    _UNIQUE_ID += 1
    return _UNIQUE_ID
