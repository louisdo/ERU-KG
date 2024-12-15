import sys, nltk
# sys.path.append("../")
from collections import Counter, OrderedDict
from src.uokg_helper.phrase_retrieval.build_index import *
from sklearn.preprocessing import normalize
from tqdm import tqdm
from src.uokg_helper.sent2vec_model import Sent2VecModel


def _knn_search(emb, index, k):
    return index.search(emb, k)

def knn_search_return_score(text, index, s2v_model, phrase_list, k, only_absent = False, thresh = None):
    text_embedding = normalize(s2v_model.embed_sentence(text), axis = 1)
    search_res = _knn_search(text_embedding, index, k)
    indices = search_res[1][0]
    distances = search_res[0][0]
    _res = [phrase_list[index] for index in indices]
    res = [[phrase, dis] for phrase, dis in zip(_res, distances)]
    if only_absent: res = [item for item in res if item[0] not in text]
    if thresh is not None: res = [item for item in res if item[1] > thresh] 
    return res

def mmr(search_results, top_k, lambda_):
    phrases = set([item[0] for item in search_results])
    phrases_tokens = {phrase:nltk.word_tokenize(phrase) for phrase in phrases}
    phrases_scores = Counter({item[0]:(item[1] + 1) / 2 for item in search_results})
    
    selected = OrderedDict()
    for i in range(top_k):
        remaining = phrases - set(selected) 
        if len(remaining) == 0: break
        mmr_score = lambda x: lambda_*phrases_scores[x] - (1-lambda_)*max([jaccard(phrases_tokens[x], phrases_tokens[y]) for y in set(list(selected.keys())[:])-{x}] or [0]) 
        next_selected = argmax(remaining, mmr_score) 
        selected[next_selected] = len(selected) 
    return selected 

def jaccard(x,y):
    tokens_x = set(x)
    tokens_y = set(y)
    return len(tokens_x.intersection(tokens_y)) / len(tokens_x.union(tokens_y))

def argmax(keys, f): 
    return max(keys, key=f) 


def knn_search(text, index, s2v_model, phrase_list, k, only_absent = False, thresh = None):
    text_embedding = normalize(s2v_model.embed_sentence(text), axis = 1)
    search_res = _knn_search(text_embedding, index, k)
    indices = search_res[1][0]
    distances = search_res[0][0]
    _res = [phrase_list[index] for index in indices]
    res = [[phrase, dis] for phrase, dis in zip(_res, distances)]
    if only_absent: res = [item for item in res if item[0] not in text]
    if thresh is not None: res = [item for item in res if item[1] > thresh] 
    return [item[0] for item in res]


def knn_search_batch(text_batch, index, s2v_model, phrase_list, k, only_absent = False, thresh = None):
    text_embeddings = normalize(s2v_model.embed_sentences(text_batch), axis = 1)
    search_res = _knn_search(text_embeddings, index, k)
    all_indices = search_res[1]
    all_distances = search_res[0]
    
    _res = [[phrase_list[index] for index in indices] for indices in all_indices]
    res = [[[phrase, dis] for phrase, dis in zip(_res[index], all_distances[index])] for index in range(len(_res))]
    if only_absent: 
        res = [[item for item in res[index] if item[0] not in text_batch[index]] for index in range(len(res))]
    if thresh is not None:
        res = [[item for item in res[index] if item[1] > thresh] for index in range(len(res))]
    return [[item[0] for item in _list] for _list in res]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out-file')
    parser.add_argument("--k", default = 25)
    parser.add_argument("--batch-size", default = 1000)
    parser.add_argument("--thresh", default = 0.7)
    args = parser.parse_args()
    
    KNN_INDEX, phrase_list = load_index("data_v2")
    print("Total number of phrases: ", len(phrase_list))
    
    SENT2VEC_CONFIG = {
        "vocab_path" : "../../data/sent2vec/vocab.json",
        "uni_embs_path": "../../data/sent2vec/uni_embs.npy"
    }
    S2V_MODEL = Sent2VecModel(SENT2VEC_CONFIG)
    
    with open("../../data/unsupervised/preprocessed_corpus_kp20k.json") as f:
        kp20k_corpus = json.load(f)[:]
        
    absent_refs = []
    for index in tqdm(range(0, len(kp20k_corpus), args.batch_size)): # batch size 100
        batch = kp20k_corpus[index:index + args.batch_size]
        absent_refs.extend(knn_search_batch(batch, KNN_INDEX, S2V_MODEL, phrase_list, args.k, only_absent = True, thresh = args.thresh))
        
    with open(args.out_file, "w") as f:
        json.dump(absent_refs, f)