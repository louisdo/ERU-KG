import os
from erukg.erukg_helper.config import HYPERPARAMS, GENERAL_CONFIG
from erukg.erukg_helper.init_models import init_splade_model, init_phraseness_module, SPLADE_MODEL, PHRASENESS_MODULE, DEVICE
from erukg.erukg_helper.splade_inference import get_tokens_scores_of_doc, get_tokens_scores_of_docs_batch, init_splade_model
from erukg.erukg_helper.utils import maybe_create_folder

index_cache_dir = os.path.join(GENERAL_CONFIG["cache_dir"], "ret_indexes")
phrase_vocab_cache_dir = os.path.join(GENERAL_CONFIG["cache_dir"], "phrase_vocab")

maybe_create_folder(index_cache_dir)
maybe_create_folder(phrase_vocab_cache_dir)