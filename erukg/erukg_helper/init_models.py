import torch, sys, os
sys.path.append("../../splade")
from transformers import AutoTokenizer
from splade.models.transformer_rep import Splade
from erukg.erukg_helper.config import MODEL_NAME_2_MODEL_INFO, GENERAL_CONFIG
from erukg.erukg_helper.retrieval_based_phraseness_module import RetrievalBasedPhrasenessModule


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLADE_MODEL = {}

PHRASENESS_MODULE = {}


def init_splade_model(model_name):
    if SPLADE_MODEL.get(model_name) is not None:
        return 
    else:
        print(f"Init splade model {model_name}. This will be done only once")

    if model_name not in MODEL_NAME_2_MODEL_INFO:
        raise NotImplementedError
    
    model_info = MODEL_NAME_2_MODEL_INFO[model_name]
    splade_dir = model_info["splade_dir"]

    print(f"Using {splade_dir}")
    
    model = Splade(splade_dir, agg="max")
    model = model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(splade_dir)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

    SPLADE_MODEL[model_name] = {
        "model": model,
        "tokenizer": tokenizer,
        "reverse_voc": reverse_voc
    }


def init_phraseness_module(model_name, neighbor_size = 100, beta = 0.8, no_retrieval = False):
    if PHRASENESS_MODULE.get(model_name) is not None:
        return
    
    print("Initializing phraseness module:", model_name)

    model_info = MODEL_NAME_2_MODEL_INFO[model_name]
    # pn_ret_index_dir = model_info["pn_ret_index_dir"]

    pn_ret_index_dir = os.path.join(GENERAL_CONFIG["cache_dir"], "ret_indexes", model_name)
    pn_ret_index_download_url = model_info["pn_ret_index_download_url"]

    PHRASENESS_MODULE[model_name] = RetrievalBasedPhrasenessModule(
        pn_ret_index_dir, 
        neighbor_size=neighbor_size, 
        document_index_download_url = pn_ret_index_download_url,
        beta = beta,
        informativeness_model_name=model_name,
        no_retrieval = no_retrieval
    )