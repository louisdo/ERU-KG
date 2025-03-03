import sys, json, faiss, os, argparse
import numpy as np
# sys.path.append("../")
# from candidate_extractor import CandidateExtractorRegExpNLTK
# from sent2vec_model import Sent2VecModel
# from utils import clean_func
from collections import Counter
from tqdm import tqdm
from sklearn.preprocessing import normalize
from multiprocessing import Pool


def extract_candidates(text):
    return CANDEXT(text)


def extract_candidates_for_corpus(corpus, CANDEXT):
    with Pool(16) as p:
        phrases_corpus = list(tqdm(p.imap(extract_candidates, corpus), desc = "Extracting phrases"))
    return phrases_corpus

def update_phrase_counter(corpus_candidates, current_counter):
    for candidates in corpus_candidates:
        current_counter.update(list(set(candidates)))
        
    return current_counter

def prune_counter(counter, thresh):
    return Counter({k:v for k,v in counter.items() if v >= thresh})


def get_corpus_embeddings(corpus, s2v_model):
    BATCH_SIZE = 100
    res = []
    for i in tqdm(range(0, len(corpus), BATCH_SIZE)):
        batch = corpus[i: i + BATCH_SIZE]
        res.append(normalize(s2v_model.embed_sentences(batch), axis = 1))
    return np.concatenate(res, axis = 0)


def build_retrieval_embedding(counter, corpus_candidates, corpus_embeddings):
    # build phrase-document index
    phrase2docs = {}
    for doc_index, candidates in enumerate(corpus_candidates):
        for c in candidates:
            if c not in counter: continue
            if c not in phrase2docs: phrase2docs[c] = []
            phrase2docs[c].append(doc_index)
            
    # build phrase inverted index
    phrase_list = list(counter.keys())
    phrase2index = {phrase:index for index,phrase in enumerate(phrase_list)}
    
    # build embedding
    retrieval_embeddings = []
    for phrase in tqdm(phrase_list):
        document_embeddings = corpus_embeddings[phrase2docs[phrase]]
        emb = np.mean(document_embeddings, axis = 0).reshape(1, -1)
        retrieval_embeddings.append(emb)
    return normalize(np.concatenate(retrieval_embeddings, axis = 0), axis = 1) , phrase_list


def knn_index(retrieval_embeddings):
    #index = faiss.IndexFlatL2(600)
    index = faiss.IndexHNSWFlat(600, 32, faiss.METRIC_INNER_PRODUCT)
    index.add(retrieval_embeddings.astype(np.float32))
    return index


def save_index(index, retrieval_embeddings, phrase_list, folder_name):
    index_file = os.path.join(folder_name, "phrase.index")
    phrase_list_file = os.path.join(folder_name, "phrase_list.json")
    retrieval_embeddings_file = os.path.join(folder_name, "context_embeddings.npy")
    
    faiss.write_index(index, index_file)
    
    with open(phrase_list_file, "w") as f:
        json.dump(phrase_list, f)
        
    with open(retrieval_embeddings_file, "wb") as f:
        np.save(f, retrieval_embeddings)
        
    print(f"Done writing index to {folder_name}")
    return True

def load_index(folder_name):
    index_file = os.path.join(folder_name, "phrase.index")
    phrase_list_file = os.path.join(folder_name, "phrase_list.json")
    
    index = faiss.read_index(index_file)
    
    with open(phrase_list_file) as f:
        phrase_list = json.load(f)
    
    return index, phrase_list



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--folder-name')
#     parser.add_argument("--thresh", default = 5)
#     args = parser.parse_args()
    
#     print("Setting up candidate extractor and sentence embedding model")
#     CANDEXT = CandidateExtractorRegExpNLTK([1,5])
#     SENT2VEC_CONFIG = {
#         "vocab_path" : "../../data/sent2vec/vocab.json",
#         "uni_embs_path": "../../data/sent2vec/uni_embs.npy"
#     }

#     S2V_MODEL = Sent2VecModel(SENT2VEC_CONFIG)

#     print("Loading corpus (KP20K, KPTimes and StackExchange)")
#     with open("../../data/unsupervised/preprocessed_corpus_kp20k.json") as f:
#         kp20k_corpus = json.load(f)[:]

#     with open("../../data/testing_datasets/KPTimes/KPTimes.train.jsonl") as f:
#         kptimes_dataset = [json.loads(line) for line in f][:]
#         kptimes_corpus = [line["title"] + ". " + line["abstract"] for line in kptimes_dataset][:]
#         del kptimes_dataset
    
#     with open("../../data/testing_datasets/StackExchange/train.json") as f:
#         stackexchange_dataset = [json.loads(line) for line in f][:]
#         stackexchange_corpus = [line["title"].lower() + ". " + line["question"].lower() for line in stackexchange_dataset]
#         del stackexchange_dataset


#     corpus = kp20k_corpus + kptimes_corpus + stackexchange_corpus
    
#     print("Extracting phrases from documents in corpus")
#     corpus_candidates = extract_candidates_for_corpus(corpus, CANDEXT)
#     # with open("../../data/unsupervised/stable/present_phrases_corpus_kp20k.json", "w") as f:
#     #     json.dump(corpus_candidates[:len(kp20k_corpus)], f)
#     # with open("../../data/unsupervised/stable/present_phrases_corpus_kptimes.json", "w") as f:
#     #     json.dump(corpus_candidates[len(kp20k_corpus):], f)

#     print("Getting phrase counter")
#     COUNTER = update_phrase_counter(corpus_candidates, Counter())
#     COUNTER = prune_counter(COUNTER, thresh = args.thresh)
#     print(f"Counter length: {len(COUNTER)}")
#     with open("../../data/unsupervised/stable/phrase_counter_kp20k_kptimes_stackexchange.json", "w") as f:
#         json.dump(COUNTER, f)
    
#     print("Get embedding vectors of documents in corpus")
#     CORPUS_EMBEDDINGS = get_corpus_embeddings(corpus, S2V_MODEL)
    
#     print("Build embedding for phrases")
#     retrieval_embeddings, phrase_list = build_retrieval_embedding(COUNTER, corpus_candidates, CORPUS_EMBEDDINGS)

#     print("Building index")
#     KNN_INDEX = knn_index(retrieval_embeddings)
    
#     print("Save index")
#     save_index(KNN_INDEX, retrieval_embeddings, phrase_list, args.folder_name)