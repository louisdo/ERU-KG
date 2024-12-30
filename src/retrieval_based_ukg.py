# import torch, sys, math
# sys.path.append("/home/lamdo/keyphrase_informativeness_test/splade")
# import numpy as np
# from typing import List, Dict
# from collections import Counter
# from transformers import AutoModelForMaskedLM, AutoTokenizer
# from splade.models.transformer_rep import Splade
# from src.retrieval_based_phraseness_module import RetrievalBasedPhrasenessModule
# from src.splade_inference import SPLADE_MODEL, get_tokens_scores_of_doc, get_tokens_scores_of_docs_batch, init_splade_model



# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device '{DEVICE}'")

# PHRASENESS_MODULE = {}


# def init_phraseness_module(model_name, neighbor_size = 100, alpha = 0.8):
#     if PHRASENESS_MODULE.get(model_name) is not None:
#         return
    
#     print("Initializing phraseness module:", model_name)
#     if model_name == "custom_trained_combined_references_v11-2":
#         path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_v11-2"
#     elif model_name == "custom_trained_combined_references_v6-2":
#         path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_v6-2"

#     elif model_name == "custom_trained_combined_references_nounphrase_v6-1":
#         path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v6-1"
#     else:
#         # for now this is the default path
#         path = "/scratch/lamdo/pyserini_experiments/index/scirepeval_search_validation_evaluation_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty"


#     PHRASENESS_MODULE[model_name] = RetrievalBasedPhrasenessModule(
#         path, 
#         neighbor_size=neighbor_size, 
#         alpha = alpha,
#         informativeness_model_name=model_name
#     )


# def is_sublist(sublist, main_list):
#     if len(sublist) > len(main_list):
#         return False
    
#     return any(sublist == main_list[i:i+len(sublist)] for i in range(len(main_list) - len(sublist) + 1))


# def merge_and_average_dicts(dict_list: List[Dict[str, float]], weights: List[float] = None):
#     if not dict_list:
#         return Counter()
    
#     # Sum up all values for each key
#     if not weights:
#         weights = [1] * len(dict_list)
#     total_weights = sum(weights)
#     combined = Counter()
#     for d, weight in zip(dict_list, weights):
#         combined.update({k: v * weight / total_weights for k, v in d.items()})
    
#     return Counter({k: v for k, v in combined.items()})


# def score_candidates_by_positions(candidates: List[str], doc: str):
#     res = Counter()
#     for cand in candidates:
#         try:
#             temp = doc.index(cand)
#             position = len([item for item in doc[:temp].split(" ") if item]) + 1
#             position_score = 1 + 1 / math.log2(position + 2) #(position + 1) / position
#         except ValueError:
#             position_score = 1
#         res[cand] = position_score
#     return res

# def score_candidates(candidates: List[str], 
#                      candidates_tokens: List[List[int]], 
#                      tokens_scores: dict,
#                      model_name: str,
#                     #  retrieved_documents_tokens_scores: List[dict] = None,
#                      retrieved_phrases_scores: List[dict] = None,
#                      retrieved_docs_scores: List[float] = None,
#                      length_penalty: int = 0,
#                      candidates_positions_scores: dict = {},
#                      candidates_phraseness_scores: dict = {},
#                      alpha = 0.8):
#     # length penalization < 0 means returning longer sequence
#     tokenized_candidates = [SPLADE_MODEL[model_name]["tokenizer"].convert_ids_to_tokens(item) for item in candidates_tokens]

#     # averaged_retrieved_documents_tokens_scores = merge_and_average_dicts(retrieved_documents_tokens_scores)
#     averaged_retrieved_phrases_scores = merge_and_average_dicts(retrieved_phrases_scores, weights = retrieved_docs_scores)
#     one_minus_alpha = 1 - alpha
#     candidates_scores = [alpha * np.sum([tokens_scores[tok] for tok in tokenized_cand]) / (len(tokenized_cand) - length_penalty) + one_minus_alpha * averaged_retrieved_phrases_scores[cand] \
#                          for tokenized_cand, cand in zip(tokenized_candidates, candidates)]

#     if candidates_phraseness_scores:
#         candidates_scores = [score * (candidates_phraseness_scores[candidates[i]] ** 1.5) for i, score in enumerate(candidates_scores)]

#     if candidates_positions_scores:
#         candidates_scores = [score * candidates_positions_scores[candidates[i]] for i, score in enumerate(candidates_scores)]
#     assert len(candidates) == len(candidates_scores)
#     return [(cand, score) for cand, score in zip(candidates, candidates_scores)]



# def keyphrase_generation(doc: str, 
#                         top_k: int = 10,
#                         informativeness_model_name: str = "",
#                         apply_position_penalty: bool = False,
#                         length_penalty: float = 0,
#                         precomputed_tokens_scores: dict = None):
    
#     init_phraseness_module(informativeness_model_name)
#     init_splade_model(informativeness_model_name)

#     lower_doc = doc.lower()
#     doc_tokens = SPLADE_MODEL[informativeness_model_name]["tokenizer"](lower_doc, return_tensors="pt", max_length = 512, truncation = True)

#     if not precomputed_tokens_scores:
#         tokens_scores = get_tokens_scores_of_doc(doc_tokens = doc_tokens, model_name = informativeness_model_name)
#     else:
#         tokens_scores = precomputed_tokens_scores


#     # candidates_phraseness_score, retrieved_documents_vectors = PHRASENESS_MODULE[informativeness_model_name](lower_doc, return_retrieved_documents_vectors=True)
#     candidates_phraseness_score, retrieved_phrases_scores = PHRASENESS_MODULE[informativeness_model_name](lower_doc, return_retrieved_phrases_scores=True)
#     candidates = list(candidates_phraseness_score.keys())

#     # candidates_tokens = [SPLADE_MODEL[informativeness_model_name]["tokenizer"].convert_ids_to_tokens(SPLADE_MODEL[informativeness_model_name]["tokenizer"](cand)["input_ids"][1:-1]) for cand in candidates]
#     candidates_tokens = [SPLADE_MODEL[informativeness_model_name]["tokenizer"](cand)["input_ids"][1:-1] for cand in candidates]

#     if apply_position_penalty:
#         candidates_positions_scores = score_candidates_by_positions(candidates, lower_doc)
#     else:
#         candidates_positions_scores = []

#     # print(candidates[0], tokens_scores[candidates[0]], candidates_positions_scores[candidates[0]])
#     scores = score_candidates(candidates,
#                               candidates_tokens,
#                               tokens_scores,
#                               model_name = informativeness_model_name, 
#                             #   retrieved_documents_tokens_scores=retrieved_documents_vectors,
#                               retrieved_phrases_scores=retrieved_phrases_scores,
#                               retrieved_docs_scores=None,
#                               length_penalty=length_penalty,
#                               candidates_positions_scores = candidates_positions_scores,
#                               candidates_phraseness_scores=candidates_phraseness_score)

#     present_indices = [i for i in range(len(candidates_tokens)) if is_sublist(candidates_tokens[i], doc_tokens["input_ids"].tolist()[0])]
#     absent_indices = [i for i in range(len(candidates_tokens)) if i not in present_indices]

#     present_candidates_scores = [scores[i] for i in present_indices]
#     present_candidates_scores = list(sorted(present_candidates_scores, key = lambda x: -x[1]))
#     absent_candidates_scores = [scores[i] for i in absent_indices]
#     absent_candidates_scores = list(sorted(absent_candidates_scores, key = lambda x: -x[1]))

#     res = {
#         "present": present_candidates_scores[:top_k],
#         "absent": absent_candidates_scores[:top_k]
#     }

#     return res


# def _keyphrase_generation_helper(
#         candidates, 
#         candidates_tokens, 
#         doc_tokens,
#         doc,
#         tokens_scores,
#         # retrieved_documents_tokens_scores,
#         retrieved_phrases_scores,
#         retrieved_docs_scores,
#         candidates_positions_scores,
#         candidates_phraseness_score,
#         informativeness_model_name, 
#         length_penalty,
#         top_k):
#     scores = score_candidates(candidates,
#                               candidates_tokens, 
#                               tokens_scores, 
#                               model_name = informativeness_model_name, 
#                             #   retrieved_documents_tokens_scores=retrieved_documents_tokens_scores,
#                               retrieved_phrases_scores = retrieved_phrases_scores,
#                               retrieved_docs_scores = retrieved_docs_scores,
#                               length_penalty=length_penalty,
#                               candidates_positions_scores = candidates_positions_scores,
#                               candidates_phraseness_scores=candidates_phraseness_score)

#     # present_indices = [i for i in range(len(candidates_tokens)) if is_sublist(candidates_tokens[i], doc_tokens)]
#     present_indices = [i for i in range(len(candidates)) if candidates[i] in doc]
#     absent_indices = [i for i in range(len(candidates_tokens)) if i not in present_indices]

#     present_candidates_scores = [scores[i] for i in present_indices]
#     present_candidates_scores = list(sorted(present_candidates_scores, key = lambda x: -x[1]))
#     absent_candidates_scores = [scores[i] for i in absent_indices]
#     absent_candidates_scores = list(sorted(absent_candidates_scores, key = lambda x: -x[1]))

#     res = {
#         "present": present_candidates_scores[:top_k],
#         "absent": absent_candidates_scores[:top_k]
#     }

#     return res


# def keyphrase_generation_batch(
#     docs: str, 
#     top_k: int = 10,
#     informativeness_model_name: str = "",
#     apply_position_penalty: bool = False,
#     length_penalty: int = 0,
#     precomputed_tokens_scores: dict = None,
#     alpha: float = 0.75):

#     init_splade_model(informativeness_model_name)
#     init_phraseness_module(informativeness_model_name)

#     PHRASENESS_MODULE[informativeness_model_name]._set_alpha(alpha)

#     lower_docs = [str(doc).lower() for doc in docs]
#     docs_tokens = SPLADE_MODEL[informativeness_model_name]["tokenizer"](lower_docs, return_tensors="pt", max_length = 512, padding = True, truncation = True)

#     if not precomputed_tokens_scores:
#         batch_tokens_scores = get_tokens_scores_of_docs_batch(docs_tokens = docs_tokens, model_name = informativeness_model_name)
#     else:
#         batch_tokens_scores = [Counter(item) for item in precomputed_tokens_scores]
#         # print("YO", batch_tokens_scores)

#     # batch_candidates_phraseness_scores, batch_retrieved_documents_vectors = PHRASENESS_MODULE[informativeness_model_name].batch_generation(docs = lower_docs, return_retrieved_documents_vectors=True)
#     batch_candidates_phraseness_scores, batch_retrieved_phrases_scores = PHRASENESS_MODULE[informativeness_model_name].batch_generation(docs = lower_docs, return_retrieved_phrases_scores=True)
#     batch_candidates = [list(candidates_phraseness_score.keys()) for candidates_phraseness_score in batch_candidates_phraseness_scores]

#     batch_candidates_tokens = [[SPLADE_MODEL[informativeness_model_name]["tokenizer"](cand)["input_ids"][1:-1] for cand in candidates] for candidates in batch_candidates]

#     if apply_position_penalty:
#         batch_candidates_positions_scores = [score_candidates_by_positions(candidates, lower_doc) for candidates, lower_doc in zip(batch_candidates, lower_docs)]
#     else:
#         batch_candidates_positions_scores = [[] for _ in range(len(batch_candidates))]
    
#     # print([i for i in range(len(_test)) if not _test[i]])
#     res = [
#         _keyphrase_generation_helper(
#             candidates, 
#             candidates_tokens, 
#             doc_tokens,
#             lower_doc,
#             tokens_scores,
#             # retrieved_documents_vectors,
#             retrieved_phrases_scores,
#             None,
#             candidates_positions_scores,
#             candidates_phraseness_score,
#             informativeness_model_name, 
#             length_penalty,
#             top_k
#         ) for candidates, candidates_tokens, doc_tokens, lower_doc, tokens_scores, retrieved_phrases_scores, candidates_positions_scores, candidates_phraseness_score in zip(
#             batch_candidates, batch_candidates_tokens, docs_tokens["input_ids"].tolist(), lower_docs, batch_tokens_scores, batch_retrieved_phrases_scores,
#             batch_candidates_positions_scores, batch_candidates_phraseness_scores
#         )
#     ]

#     return res


import torch, sys, math
sys.path.append("/home/lamdo/keyphrase_informativeness_test/splade")
import numpy as np
from typing import List, Dict
from collections import Counter
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade
from src.retrieval_based_phraseness_module import RetrievalBasedPhrasenessModule
from src.splade_inference import SPLADE_MODEL, get_tokens_scores_of_doc, get_tokens_scores_of_docs_batch, init_splade_model



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device '{DEVICE}'")

PHRASENESS_MODULE = {}


def init_phraseness_module(model_name, neighbor_size = 100, alpha = 0.8):
    if PHRASENESS_MODULE.get(model_name) is not None:
        return
    
    print("Initializing phraseness module:", model_name)
    if model_name == "custom_trained_combined_references_v11-2":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_v11-2"
    elif model_name == "custom_trained_combined_references_v6-2":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_v6-2"

    elif model_name == "custom_trained_combined_references_nounphrase_v6-1":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v6-1"
    elif model_name == "custom_trained_combined_references_nounphrase_v6-2":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v6-2"
    elif model_name == "custom_trained_combined_references_nounphrase_v6-3":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v6-3"
    elif model_name == "custom_trained_combined_references_nounphrase_v6-4":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v6-4"
    elif model_name == "custom_trained_combined_references_nounphrase_v6-5":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v6-5"


    elif model_name == "custom_trained_combined_references_nounphrase_v7-1":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v7-1"
    elif model_name == "custom_trained_combined_references_nounphrase_v7-2":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v7-2"
    elif model_name == "custom_trained_combined_references_nounphrase_v7-3":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v7-3"
    elif model_name == "custom_trained_combined_references_nounphrase_v7-4":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v7-4"
    elif model_name == "custom_trained_combined_references_nounphrase_v7-5":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v7-5"

    elif model_name == "custom_trained_combined_references_no_titles_nounphrase_v6-1":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_no_titles_v6-1"
    elif model_name == "custom_trained_combined_references_no_queries_nounphrase_v6-1":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_no_queries_v6-1"
    elif model_name == "custom_trained_combined_references_no_cc_nounphrase_v6-1":
        path = "/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_no_cc_v6-1"
    else:
        # for now this is the default path
        path = "/scratch/lamdo/pyserini_experiments/index/scirepeval_search_validation_evaluation_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty"


    PHRASENESS_MODULE[model_name] = RetrievalBasedPhrasenessModule(
        path, 
        neighbor_size=neighbor_size, 
        alpha = alpha,
        informativeness_model_name=model_name
    )


def is_sublist(sublist, main_list):
    if len(sublist) > len(main_list):
        return False
    
    return any(sublist == main_list[i:i+len(sublist)] for i in range(len(main_list) - len(sublist) + 1))


def merge_and_average_dicts(dict_list: List[Dict[str, float]], weights: List[float] = None):
    if not dict_list:
        return Counter()
    
    if not weights:
        weights = [1 / len(dict_list)] * len(dict_list)
    
    # Sum up all values for each key
    combined = Counter()
    for d, weight in zip(dict_list, weights):
        combined.update({k: v * weight for k,v in d.items()})
    
    # Divide by the number of dictionaries
    # num_dicts = len(dict_list)
    # return Counter({k: v / num_dicts for k, v in combined.items()})
    return Counter(combined)


def score_candidates_by_positions(candidates: List[str], doc: str):
    res = Counter()
    for cand in candidates:
        try:
            temp = doc.index(cand)
            position = len([item for item in doc[:temp].split(" ") if item]) + 1
            position_score = 1 + 1 / math.log2(position + 2) #(position + 1) / position
        except ValueError:
            position_score = 1
        res[cand] = position_score
    return res

def score_candidates(candidates: List[str], 
                     candidates_tokens: List[List[int]], 
                     tokens_scores: dict,
                     model_name: str,
                     retrieved_documents_tokens_scores: List[dict] = None,
                     retrieved_documents_scores: List[float] = None, # this is the retrieval scores of the retrieved documents
                     length_penalty: int = 0,
                     candidates_positions_scores: dict = {},
                     candidates_phraseness_scores: dict = {},
                     alpha = 0.8):
    # length penalization < 0 means returning longer sequence
    tokenized_candidates = [SPLADE_MODEL[model_name]["tokenizer"].convert_ids_to_tokens(item) for item in candidates_tokens]

    averaged_retrieved_documents_tokens_scores = merge_and_average_dicts(retrieved_documents_tokens_scores, weights = retrieved_documents_scores)
    one_minus_alpha = 1 - alpha
    candidates_scores = [np.sum([alpha * tokens_scores[tok] + one_minus_alpha * averaged_retrieved_documents_tokens_scores[tok] for tok in tokenized_cand]) / (len(tokenized_cand) - length_penalty) for tokenized_cand in tokenized_candidates]

    if candidates_phraseness_scores:
        candidates_scores = [score * (candidates_phraseness_scores[candidates[i]] ** 1.5) for i, score in enumerate(candidates_scores)]

    if candidates_positions_scores:
        candidates_scores = [score * candidates_positions_scores[candidates[i]] for i, score in enumerate(candidates_scores)]
    assert len(candidates) == len(candidates_scores)
    return [(cand, score) for cand, score in zip(candidates, candidates_scores)]



def keyphrase_generation(doc: str, 
                        top_k: int = 10,
                        informativeness_model_name: str = "",
                        apply_position_penalty: bool = False,
                        length_penalty: float = 0,
                        precomputed_tokens_scores: dict = None):
    
    init_phraseness_module(informativeness_model_name)
    init_splade_model(informativeness_model_name)

    lower_doc = doc.lower()
    doc_tokens = SPLADE_MODEL[informativeness_model_name]["tokenizer"](lower_doc, return_tensors="pt", max_length = 512, truncation = True)

    if not precomputed_tokens_scores:
        tokens_scores = get_tokens_scores_of_doc(doc_tokens = doc_tokens, model_name = informativeness_model_name)
    else:
        tokens_scores = precomputed_tokens_scores

    phraseness_module_output = PHRASENESS_MODULE[informativeness_model_name](
        doc, return_retrieved_documents_vectors=True, return_retrieved_documents_scores=True)
    
    candidates_phraseness_score = phraseness_module_output["keyphrase_candidates_scores"]
    retrieved_documents_vectors = phraseness_module_output["retrieved_documents_vectors"]
    retrieved_documents_scores = phraseness_module_output["retrieved_documents_scores"]

    candidates = list(candidates_phraseness_score.keys())

    # candidates_tokens = [SPLADE_MODEL[informativeness_model_name]["tokenizer"].convert_ids_to_tokens(SPLADE_MODEL[informativeness_model_name]["tokenizer"](cand)["input_ids"][1:-1]) for cand in candidates]
    candidates_tokens = [SPLADE_MODEL[informativeness_model_name]["tokenizer"](cand)["input_ids"][1:-1] for cand in candidates]

    if apply_position_penalty:
        candidates_positions_scores = score_candidates_by_positions(candidates, lower_doc)
    else:
        candidates_positions_scores = []

    # print(candidates[0], tokens_scores[candidates[0]], candidates_positions_scores[candidates[0]])
    scores = score_candidates(candidates,
                              candidates_tokens,
                              tokens_scores,
                              model_name = informativeness_model_name, 
                              retrieved_documents_tokens_scores=retrieved_documents_vectors,
                              retrieved_documents_scores=retrieved_documents_scores,
                              length_penalty=length_penalty,
                              candidates_positions_scores = candidates_positions_scores,
                              candidates_phraseness_scores=candidates_phraseness_score)

    # present_indices = [i for i in range(len(candidates_tokens)) if is_sublist(candidates_tokens[i], doc_tokens["input_ids"].tolist()[0])]
    # absent_indices = [i for i in range(len(candidates_tokens)) if i not in present_indices]

    present_indices = [i for i in range(len(candidates)) if candidates[i] in lower_doc]
    absent_indices = [i for i in range(len(candidates_tokens)) if i not in present_indices]

    present_candidates_scores = [scores[i] for i in present_indices]
    present_candidates_scores = list(sorted(present_candidates_scores, key = lambda x: -x[1]))
    absent_candidates_scores = [scores[i] for i in absent_indices]
    absent_candidates_scores = list(sorted(absent_candidates_scores, key = lambda x: -x[1]))

    res = {
        "present": present_candidates_scores[:top_k],
        "absent": absent_candidates_scores[:top_k]
    }

    return res


def _keyphrase_generation_helper(
        candidates, 
        candidates_tokens, 
        doc_tokens,
        lower_doc,
        tokens_scores,
        retrieved_documents_tokens_scores,
        retrieved_documents_scores, # this is the retrieval scores of the retrieved documents
        candidates_positions_scores,
        candidates_phraseness_score,
        informativeness_model_name, 
        length_penalty,
        top_k):
    scores = score_candidates(candidates,
                              candidates_tokens, 
                              tokens_scores, 
                              model_name = informativeness_model_name, 
                              retrieved_documents_tokens_scores=retrieved_documents_tokens_scores,
                              retrieved_documents_scores = retrieved_documents_scores,
                              length_penalty=length_penalty,
                              candidates_positions_scores = candidates_positions_scores,
                              candidates_phraseness_scores=candidates_phraseness_score)

    # present_indices = [i for i in range(len(candidates_tokens)) if is_sublist(candidates_tokens[i], doc_tokens)]
    present_indices = [i for i in range(len(candidates)) if candidates[i] in lower_doc]
    absent_indices = [i for i in range(len(candidates_tokens)) if i not in present_indices]

    present_candidates_scores = [scores[i] for i in present_indices]
    present_candidates_scores = list(sorted(present_candidates_scores, key = lambda x: -x[1]))
    absent_candidates_scores = [scores[i] for i in absent_indices]
    absent_candidates_scores = list(sorted(absent_candidates_scores, key = lambda x: -x[1]))

    res = {
        "present": present_candidates_scores[:top_k],
        "absent": absent_candidates_scores[:top_k]
    }

    return res


def keyphrase_generation_batch(
    docs: str, 
    top_k: int = 10,
    informativeness_model_name: str = "",
    apply_position_penalty: bool = False,
    length_penalty: int = 0,
    precomputed_tokens_scores: dict = None,
    alpha: float = 0.75):

    init_splade_model(informativeness_model_name)
    init_phraseness_module(informativeness_model_name)

    PHRASENESS_MODULE[informativeness_model_name]._set_alpha(alpha)

    lower_docs = [str(doc).lower() for doc in docs]
    docs_tokens = SPLADE_MODEL[informativeness_model_name]["tokenizer"](lower_docs, return_tensors="pt", max_length = 512, padding = True, truncation = True)

    if not precomputed_tokens_scores:
        batch_tokens_scores = get_tokens_scores_of_docs_batch(docs_tokens = docs_tokens, model_name = informativeness_model_name)
    else:
        batch_tokens_scores = [Counter(item) for item in precomputed_tokens_scores]
        # print("YO", batch_tokens_scores)

    phraseness_module_output = PHRASENESS_MODULE[informativeness_model_name].batch_generation(
        docs = docs, return_retrieved_documents_vectors=True, return_retrieved_documents_scores = True)
    
    batch_candidates_phraseness_scores = phraseness_module_output["keyphrase_candidates_scores"]
    batch_retrieved_documents_vectors = phraseness_module_output["retrieved_documents_vectors"]
    batch_retrieved_documents_scores = phraseness_module_output["retrieved_documents_scores"]

    batch_candidates = [list(candidates_phraseness_score.keys()) for candidates_phraseness_score in batch_candidates_phraseness_scores]

    batch_candidates_tokens = [[SPLADE_MODEL[informativeness_model_name]["tokenizer"](cand)["input_ids"][1:-1] for cand in candidates] for candidates in batch_candidates]

    if apply_position_penalty:
        batch_candidates_positions_scores = [score_candidates_by_positions(candidates, lower_doc) for candidates, lower_doc in zip(batch_candidates, lower_docs)]
    else:
        batch_candidates_positions_scores = [[] for _ in range(len(batch_candidates))]
    
    # print([i for i in range(len(_test)) if not _test[i]])
    res = [
        _keyphrase_generation_helper(
            candidates, 
            candidates_tokens, 
            doc_tokens,
            lower_doc,
            tokens_scores,
            retrieved_documents_vectors,
            retrieved_documents_scores,
            candidates_positions_scores,
            candidates_phraseness_score,
            informativeness_model_name, 
            length_penalty,
            top_k
        ) for candidates, candidates_tokens, doc_tokens, lower_doc, tokens_scores, retrieved_documents_vectors, retrieved_documents_scores, candidates_positions_scores, candidates_phraseness_score in zip(
            batch_candidates, batch_candidates_tokens, docs_tokens["input_ids"].tolist(), lower_docs, batch_tokens_scores, batch_retrieved_documents_vectors, batch_retrieved_documents_scores,
            batch_candidates_positions_scores, batch_candidates_phraseness_scores
        )
    ]

    return res