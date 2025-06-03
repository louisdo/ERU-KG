import torch, sys, traceback
sys.path.append("../splade")
from collections import Counter
from erukg.erukg_helper.init_models import SPLADE_MODEL, DEVICE, init_splade_model


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