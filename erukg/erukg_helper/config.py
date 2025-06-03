import os
from pathlib import Path

HYPERPARAMS = {
    "alpha": 0.8,
    "beta": 0.8,
    "lambda": 1.5,
    "length_penalty": -0.25,
    "neighbor_size": 100,
}

cache_dir = os.getenv("ERUKG_CACHE_DIR")
default_cache_dir = os.path.join(str(Path.home()), ".cache", "erukg_cache")
GENERAL_CONFIG = {
    "cache_dir": cache_dir if cache_dir else default_cache_dir
}


MODEL_NAME_2_MODEL_INFO = {
    "eru-kg-base": {"splade_dir": "lamdo/eru-kg-base", 
                    "pn_ret_index_download_url": "https://drive.google.com/file/d/15m3Ofsw_MHe0tbiDpNQU8TOpcoriIOvp/view?usp=drive_link"},
    "eru-kg-small": {"splade_dir": "lamdo/eru-kg-small", 
                     "pn_ret_index_download_url": "https://drive.google.com/file/d/1g_OzVy8TbiYjuFey8925FXz48ewcHsil/view?usp=sharing"}
}