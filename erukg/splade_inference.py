import torch, sys, traceback
sys.path.append("/home/lamdo/keyphrase_informativeness_test/splade")
from transformers import AutoModelForMaskedLM, AutoTokenizer
from collections import Counter
from splade.models.transformer_rep import Splade


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
    elif model_name == "custom_trained_combined_references_v11-1":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v11-1/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v11-2":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v11-2/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v11-3":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v11-3/debug/checkpoint/model"

    elif model_name == "custom_trained_combined_references_v6-1" or model_name == "custom_trained_combined_references_nounphrase_v6-1":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v6-1/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v6-2" or model_name == "custom_trained_combined_references_nounphrase_v6-2":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v6-2/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v6-3" or model_name == "custom_trained_combined_references_nounphrase_v6-3":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v6-3/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v6-4" or model_name == "custom_trained_combined_references_nounphrase_v6-4":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v6-4/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v6-5" or model_name == "custom_trained_combined_references_nounphrase_v6-5":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v6-5/debug/checkpoint/model"

    
    elif model_name == "custom_trained_combined_references_v7-1" or model_name == "custom_trained_combined_references_nounphrase_v7-1":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v7-1/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v7-2" or model_name == "custom_trained_combined_references_nounphrase_v7-2":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v7-2/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v7-3" or model_name == "custom_trained_combined_references_nounphrase_v7-3":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v7-3/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v7-4" or model_name == "custom_trained_combined_references_nounphrase_v7-4":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v7-4/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v7-5" or model_name == "custom_trained_combined_references_nounphrase_v7-5":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v7-5/debug/checkpoint/model"


    elif model_name == "custom_trained_combined_references_no_titles_v6-1" or model_name == "custom_trained_combined_references_no_titles_nounphrase_v6-1":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_no_titles_v6-1/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_no_cc_v6-1" or model_name == "custom_trained_combined_references_no_cc_nounphrase_v6-1":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_no_cc_v6-1/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_no_queries_v6-1" or model_name == "custom_trained_combined_references_no_queries_nounphrase_v6-1":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_no_queries_v6-1/debug/checkpoint/model"


    elif model_name == "custom_trained_combined_references_v8-1" or model_name == "custom_trained_combined_references_nounphrase_v8-1":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v8-1/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v8-2" or model_name == "custom_trained_combined_references_nounphrase_v8-2":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v8-2/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v8-3" or model_name == "custom_trained_combined_references_nounphrase_v8-3":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v8-3/debug/checkpoint/model"


    elif model_name == "custom_trained_combined_references_v9-1" or model_name == "custom_trained_combined_references_nounphrase_v9-1":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v9-1/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v9-2" or model_name == "custom_trained_combined_references_nounphrase_v9-2":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v9-2/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_v9-3" or model_name == "custom_trained_combined_references_nounphrase_v9-3":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v9-3/debug/checkpoint/model"

    elif model_name == "custom_trained_combined_references_no_titles_v8-1" or model_name == "custom_trained_combined_references_no_titles_nounphrase_v8-1":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_no_titles_v8-1/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_no_cc_v8-1" or model_name == "custom_trained_combined_references_no_cc_nounphrase_v8-1":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_no_cc_v8-1/debug/checkpoint/model"
    elif model_name == "custom_trained_combined_references_no_queries_v8-1" or model_name == "custom_trained_combined_references_no_queries_nounphrase_v8-1":
        model_type_or_dir = "/scratch/lamdo/splade_checkpoints/experiments_combined_references_no_queries_v8-1/debug/checkpoint/model"
        
    else: raise NotImplementedError

    

    print(f"Using {model_type_or_dir}")

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

def get_tokens_scores_of_doc(doc_tokens, model_name):
    try:
        # now compute the document representation
        with torch.no_grad():
            doc_rep = SPLADE_MODEL[model_name]["model"](d_kwargs=doc_tokens.to(DEVICE))["d_rep"].squeeze().cpu()  # (sparse) doc rep in voc space, shape (30522,)

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



def get_tokens_scores_of_docs_batch(docs_tokens, model_name):
    # try:
    # Compute the document representations for the batch
    with torch.no_grad():
        docs_rep = SPLADE_MODEL[model_name]["model"](d_kwargs=docs_tokens.to(DEVICE))["d_rep"].cpu()  # shape (batch_size, 30522)

    batch_results = []
    
    for doc_rep in docs_rep:
        try:
            # Get the number of non-zero dimensions in the rep:
            col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
            
            # Get the weights for non-zero dimensions
            weights = doc_rep[col].cpu().tolist()

            if not isinstance(weights, list) and not isinstance(col, list):
                weights = [weights]
                col = [col]
            
            # Create a dictionary of dimension indices and their weights
            d = {k: v for k, v in zip(col, weights)}
            sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
            
            # Create the BOW representation
            bow_rep = []
            for k, v in sorted_d.items():
                bow_rep.append((SPLADE_MODEL[model_name]["reverse_voc"][k], round(v, 2)))
            
            # Add the Counter object to the batch_results
            batch_results.append(Counter({line[0]: line[1] for line in bow_rep}))
        except Exception as e:
            print("Error in getting token score (informativeness module)", traceback.format_exc(), col, weights)
            batch_results.append(Counter())

    return batch_results
    # except Exception as e:
    #     print("Error in getting token score (informativeness module)", traceback.format_exc(), col, weights, )
    #     return [Counter() for _ in range(len(docs_tokens.input_ids))]