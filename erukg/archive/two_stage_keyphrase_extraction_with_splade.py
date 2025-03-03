import torch, sys, math
sys.path.append("/home/lamdo/keyphrase_informativeness_test/splade")
import numpy as np
from collections import Counter
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade
from erukg.nounphrase_extractor import CandidateExtractorRegExpNLTK



# set the dir for trained weights

##### v2
# model_type_or_dir = "naver/splade_v2_max"
# model_type_or_dir = "naver/splade_v2_distil"

### v2bis, directly download from Hugging Face
# model_type_or_dir = "naver/splade-cocondenser-selfdistil"
# model_type_or_dir = "naver/splade-cocondenser-ensembledistil"
# model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments/debug/checkpoint/model"


# loading model and tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device '{DEVICE}'")

CANDEXT = CandidateExtractorRegExpNLTK([1,5])


SPLADE_MODEL = {}

def init_splade_model(model_name):
    if SPLADE_MODEL.get(model_name) is not None:
        return 
    else:
        print(f"Init splade model {model_name}. This will be done only once")

    if model_name == "custom_trained_pubmedqa+specter":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_pubmedqa+specter/debug/checkpoint/model"

    elif model_name == "custom_trained_msmarco":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_msmarco/debug/checkpoint/model"

    elif model_name == "splade-cocondenser-ensembledistil":
        model_type_or_dir = "naver/splade-cocondenser-ensembledistil"

    elif model_name == "splade-cocondenser-selfdistil":
        model_type_or_dir = "naver/splade-cocondenser-selfdistil"
    elif model_name == "custom_trained_scirepeval_search":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_scirepeval/debug/checkpoint/model"
    elif model_name == "custom_trained_scirepeval_search_v2":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_scirepeval_search_v2/debug/checkpoint/model"
    elif model_name == "custom_trained_scirepeval_highly_influential_citation":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_scirepeval_high_influential_citation/debug/checkpoint/model"
    elif model_name == "custom_trained_scirepeval_highly_influential_citation->search":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_scirepeval_high_influential_citation->search/debug/checkpoint/model"
    elif model_name == "custom_trained_scirepeval_highly_influential_citation+search":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_scirepeval_high_influential_citation+search/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_citation_context":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_citation_context->scirepeval_search":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive--scirepeval_search/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_citation_context+scirepeval_search":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive+scirepeval_search/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxiv_citation_context_random_negsampling_only":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxiv_citation_context_random_negsampling_only_2/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxiv_intro_related_work":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive_intro_related_work/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_full_paper":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive_full_paper/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_full_paper_v2":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive_full_paper_v2/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_full_paper->scirepeval_search":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive_full_paper->scirepeval_search/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_full_paper_v2":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive_full_paper_v2/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_full_paper_v2->scirepeval_search_v2":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive_full_paper_v2->scirepeval_search_v2/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_intro_relatedwork_1citationpersentence":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive_intro_relatedwork_1citationpersentence/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_intro_relatedwork_1citationpersentence->scirepeval_search":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive_intro_relatedwork_1citationpersentence->scirepeval_search/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_full_paper_1citationpersentence":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive_full_paper_1citationpersentence/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_full_paper_1citationpersentence->scirepeval_search":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive_full_paper_1citationpersentence->scirepeval_search/debug/checkpoint/model"
    elif model_name == "custom_trained_splade_doc_unarxive_full_paper_1citationpersentence":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_splade_doc_unarxive_full_paper_1citationpersentence/debug/checkpoint/model"
    elif model_name == "custom_trained_splade_doc_unarxive_full_paper_1citationpersentence->scirepeval_search":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_splade_doc_unarxive_full_paper_1citationpersentence->scirepeval_search/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search/debug/checkpoint/model"
    elif model_name == "custom_trained_scirepeval_search->unarxive_intro_relatedwork_1citationpersentence":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_scirepeval_search->unarxive_intro_relatedwork_1citationpersentence/debug/checkpoint/model"
    elif model_name == "custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search_v2":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search_v2/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v2":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v2/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v3":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v3/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v4":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v4/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v5":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v5/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v6":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v6/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v7":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v7/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v8":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v8/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v9":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v9/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v10-1":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v10-1/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v11-1":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v11-1/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v11-2":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v11-2/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v11-3":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v11-3/debug/checkpoint/model"
    else: raise NotImplementedError

    # if model_name in ["custom_trained_combined_references_v9"]:
    #     model = SpladeSoftPlus(model_type_or_dir, agg="max")
    # else:
    
    model = Splade(model_type_or_dir, agg="max")
    model = model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

    SPLADE_MODEL[model_name] = {
        "model": model,
        "tokenizer": tokenizer,
        "reverse_voc": reverse_voc
    }



def get_tokens_scores_of_doc(doc, model_name):
    try:
        # now compute the document representation
        with torch.no_grad():
            doc_rep = SPLADE_MODEL[model_name]["model"](d_kwargs=SPLADE_MODEL[model_name]["tokenizer"](doc, return_tensors="pt", max_length = 512, truncation = True).to(DEVICE))["d_rep"].squeeze().cpu()  # (sparse) doc rep in voc space, shape (30522,)

        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
        # print("number of actual dimensions: ", len(col))

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        d = {k: v for k, v in zip(col, weights)}
        sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
        bow_rep = []
        for k, v in sorted_d.items():
            bow_rep.append((SPLADE_MODEL[model_name]["reverse_voc"][k], round(v, 2)))
        # print("SPLADE BOW rep:\n", bow_rep)

        return Counter({line[0]: line[1] for line in bow_rep})
    except Exception as e:
        print(e)
        return Counter()


def score_candidates_by_positions(candidates, doc):
    res = Counter()
    for cand in candidates:
        try:
            temp = doc.index(cand)
            position = len([item for item in doc[:temp].split(" ") if item]) + 1
            position_score = (position + 1) / position
        except ValueError:
            position_score = 0
        res[cand] = position_score
    return res

def score_candidates(candidates, 
                     tokens_scores, 
                     model_name, 
                     length_penalty = 0,
                     candidates_positions_scores = {}):
    # length penalization < 0 means returning longer sequence
    tokenized_candidates = [SPLADE_MODEL[model_name]["tokenizer"].convert_ids_to_tokens(SPLADE_MODEL[model_name]["tokenizer"](cand)["input_ids"][1:-1]) for cand in candidates]

    candidates_scores = [np.sum([tokens_scores[tok] for tok in tokenized_cand]) / (len(tokenized_cand) - length_penalty) for tokenized_cand in tokenized_candidates]

    if candidates_positions_scores:
        candidates_scores = [score * candidates_positions_scores[candidates[i]] for i, score in enumerate(candidates_scores)]
    assert len(candidates) == len(candidates_scores)
    return [(cand, score) for cand, score in zip(candidates, candidates_scores)]


def keyphrase_extraction(doc, 
                         top_k = 10, 
                         model_name = "splade-cocondenser-ensembledistil",
                         apply_position_penalty = False,
                         length_penalty = 0,
                         precomputed_tokens_scores = None):
    init_splade_model(model_name)

    lower_doc = doc.lower()

    if not precomputed_tokens_scores:
        tokens_scores = get_tokens_scores_of_doc(doc, model_name = model_name)
    else:
        tokens_scores = precomputed_tokens_scores
        
    candidates = CANDEXT(lower_doc)

    if apply_position_penalty:
        candidates_positions_scores = score_candidates_by_positions(candidates, lower_doc)
    else:
        candidates_positions_scores = []

    # print(candidates[0], tokens_scores[candidates[0]], candidates_positions_scores[candidates[0]])
    scores = score_candidates(candidates, 
                              tokens_scores, 
                              model_name = model_name, 
                              length_penalty=length_penalty,
                              candidates_positions_scores = candidates_positions_scores)

    res = {
        "present": list(sorted(scores, key = lambda x: -x[1]))[:top_k],
        "absent": []
    }
    return res