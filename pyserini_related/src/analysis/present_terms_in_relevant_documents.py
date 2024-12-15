import os, nltk, json, string, re
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from tqdm import tqdm
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')

DATASET_NAME = os.environ["DATASET_NAME"]

STEMMER = PorterStemmer()

try:
    os.remove("logging.txt")
except Exception:
    pass


def simple_tokenize(text: str, lower = False):
    if not text: return []
    if not isinstance(text, str): text = str(text)
    if lower:
        res = [
            tok.strip(string.punctuation).strip("\n").lower() for tok in re.split(r"[-\,\(\)\s]+", text)
        ]
    else:
        res = [tok.strip(string.punctuation).strip("\n") for tok in re.split(r"[-\,\(\)\s]+", text)]

    return [tok for tok in res if tok]

def check_query_terms_in_doc(query, doc):
    doc_terms = simple_tokenize(doc)
    query_terms = simple_tokenize(query)
    res = [qterm for qterm in query_terms if qterm in doc_terms]

    with open("logging.txt", "a") as f:
        f.write(str([res, query_terms]))
        f.write("\n")

    return len(res) / len(query_terms)

def check_query_terms_in_doc_v2(query, doc):
    query_terms = simple_tokenize(query)
    query_terms = [qterm for qterm in query_terms if qterm not in STOPWORDS]
    res = [qterm for qterm in query_terms if qterm in doc]

    with open("logging.txt", "a") as f:
        f.write(str([res, query_terms]))
        f.write("\n")

    return len(res) / len(query_terms), res

def stem_text(text):
    # Download the punkt tokenizer if not already downloaded
    nltk.download('punkt', quiet=True)
    
    
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Stem each word
    stemmed_words = [STEMMER.stem(word) for word in words]
    
    # Join the stemmed words back into a single string
    stemmed_text = ' '.join(stemmed_words)
    
    return stemmed_text


def analysis_of_absence_of_query_terms(dataset_name):
    logging_for_debugging = []
    if dataset_name == "scifact":
        ds_qrels = load_dataset("BeIR/scifact-qrels")

        ds_corpus = load_dataset("BeIR/scifact", "corpus")
        ds_queries = load_dataset("BeIR/scifact", "queries")

        corpus = {}
        for line in tqdm(ds_corpus["corpus"]):
            jline = line
            
            docid = jline["_id"]
            title = jline["title"]
            abstract = jline["text"]
            text = f"{title}. {abstract}"
            corpus[str(docid)] = text.lower() #stem_text(text.lower())

        queries = {}
        for line in tqdm(ds_queries["queries"]):
            jline = line
            
            docid = jline["_id"]
            title = jline["title"]
            abstract = jline["text"]
            text = f"{title}. {abstract}"
            queries[str(docid)] = text.lower() #stem_text(text.lower())

        qid2docids = {}
        for split in ["train", "test"]:
            for line in ds_qrels[split]:
                qid = str(line["query-id"])
                docid = str(line["corpus-id"]).replace(",", "")
                score = line["score"]

                # if score < 1: continue

                if qid not in qid2docids: qid2docids[qid] = []
                if score > 0:
                    qid2docids[qid].append({
                        "docid": docid,
                        "score": score
                    })

    elif dataset_name == "trec_covid":
        ds_qrels = load_dataset("BeIR/trec-covid-qrels")

        ds_corpus = load_dataset("BeIR/trec-covid", "corpus")
        ds_queries = load_dataset("BeIR/trec-covid", "queries")

        corpus = {}
        for line in tqdm(ds_corpus["corpus"]):
            jline = line
            
            docid = jline["_id"]
            title = jline["title"]
            abstract = jline["text"]
            text = f"{title}. {abstract}"
            corpus[str(docid)] = text.lower() #stem_text(text.lower())

        queries = {}
        for line in tqdm(ds_queries["queries"]):
            jline = line
            
            docid = jline["_id"]
            title = jline["title"]
            abstract = jline["text"]
            text = f"{title}. {abstract}"
            queries[str(docid)] = text.lower() #stem_text(text.lower())

        qid2docids = {}
        for split in ["test"]:
            for line in ds_qrels[split]:
                qid = str(line["query-id"])
                docid = str(line["corpus-id"]).replace(",", "")
                score = line["score"]

                # if score < 1: continue

                if qid not in qid2docids: qid2docids[qid] = []
                if score > 0:
                    qid2docids[qid].append({
                        "docid": docid,
                        "score": score
                    })

    elif dataset_name == "scirepeval_search":
        ds = load_dataset("allenai/scirepeval", "search")

        ds_evaluation = ds["evaluation"]

        queries = {}
        qid2docids = {}
        for line in ds_evaluation:
            line_query = line.get("query")
            candidates = line.get("candidates")
            query_id = line.get("doc_id")

            lowered_line_query = line_query.lower()
            candidates_titles = [c.get("title").lower() for c in candidates]

            # this is to remove easy cases where the query is the title of some paper
            # can comment this if needed
            if any([lowered_line_query in ctitle for ctitle in candidates_titles]): continue


            # line_labels = [str(item["doc_id"]) for item in candidates if item["score"] > 0]
            line_labels = [{"docid": str(item["doc_id"]), "score": item["score"]} for item in candidates]
            # if not line_labels: number_of_queries_with_no_rel_docs += 1

            queries[str(query_id)] = line_query.lower()
            qid2docids[query_id] = [item for item in line_labels if item["score"] > 0]

        corpus = {}
        with open("/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search_validation_evaluation.jsonl") as f:
            for line in tqdm(f):
                jline = json.loads(line)
                
                # title = jline["title"]
                # abstract = jline["abstract"]
                # text = f"{title}\n{abstract}"

                doc_id = jline["doc_id"]
                corpus[str(doc_id)] = line.lower()


    all_query_presence = []
    all_query_presence_max = []

    for qid in tqdm(qid2docids):
        query = queries[qid]
        query_presence_max = 0
        for line in qid2docids[qid]:
            docid = line.get("docid")
            if not docid: continue
            doc = corpus[docid]

            query_overlap, query_terms_present_in_doc = check_query_terms_in_doc_v2(query, doc)
            if query_overlap < 1:
                logging_for_debugging.append([query, query_terms_present_in_doc, doc])
            all_query_presence.append(query_overlap)
            if query_overlap > query_presence_max: query_presence_max = query_overlap
        
        all_query_presence_max.append(query_presence_max)
    
    with open(f"logging_for_debugging_{dataset_name}.json", "w") as f:
        json.dump(logging_for_debugging, f, indent = 4)
    return all_query_presence, all_query_presence_max


if __name__ == "__main__":
    all_query_presence, all_query_presence_max = analysis_of_absence_of_query_terms(DATASET_NAME)

    print(np.mean(all_query_presence), np.mean(all_query_presence_max))