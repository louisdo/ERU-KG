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
    "cache_dir": cache_dir if cache_dir else default_cache_dir,
    "inference_max_batch_size": 128
}


MODEL_NAME_2_MODEL_INFO = {
    "eru-kg-base": {"splade_dir": "lamdo/eru-kg-base", 
                    "pn_ret_index_download_url": "https://zenodo.org/records/17109453/files/eru-kg-base.zip"},
    "eru-kg-small": {"splade_dir": "lamdo/eru-kg-small", 
                     "pn_ret_index_download_url": "https://zenodo.org/records/17109453/files/eru-kg-small.zip"}
}