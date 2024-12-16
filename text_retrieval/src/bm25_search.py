import os, json
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
from evaluation.genret_eval_scripts import eval_recall, eval_ndcg, eval_mrr
from datetime import datetime
from datasets import load_dataset


INDEX_PATH = os.environ["INDEX_PATH"]
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")

GROUNDTRUTH_DATA_PATH = os.getenv("GROUNDTRUTH_DATA_PATH")
DATASET_NAME = os.getenv("DATASET_NAME")

SEARCHER = LuceneSearcher(INDEX_PATH)

def get_current_date_string():
    current_date = datetime.now()
    return current_date.strftime("%d %B %Y")

def search_single_query(query, searcher = SEARCHER, top_k = 50, return_raw = False):
    if not query: return []
    hits = searcher.search(query, k = top_k)

    res = []
    for line in hits:
        docid = line.docid
        score = line.score

        new_line = {
            "docid": docid,
            "score": score
        }
        if return_raw:
            raw = line.lucene_document.get('raw')
            new_line["raw"] = raw

        res.append(new_line)

    return res


def search_multiple_queries(queries, 
                            searcher = SEARCHER, 
                            top_k = 50,
                            show_progress_bar = False):
    pbar = queries if not show_progress_bar else tqdm(queries)

    all_search_results = []
    for query in pbar:
        search_results = search_single_query(query = query, searcher = searcher, top_k=top_k)

        all_search_results.append(search_results)
    
    return all_search_results


def do_search_and_evaluate(queries, all_labels,
                           searcher = SEARCHER, 
                           top_k = 50,
                           show_progress_bar = False):
    
    all_search_results = search_multiple_queries(queries=queries, searcher=searcher, top_k=top_k, show_progress_bar=show_progress_bar)

    predictions = [[str(item.get("docid")) for item in line] for line in all_search_results]

    print(all_labels[0], predictions[0])

    result = {}
    for eval_top_k in [10, 100]:
        eval_results_recall = eval_recall(predict=predictions, label = all_labels, at = eval_top_k)
        eval_results_ndcg = eval_ndcg(predict = predictions, label = all_labels, at = eval_top_k)
        eval_results_mrr = eval_mrr(predict = predictions, label = all_labels, at = eval_top_k)
        result.update(eval_results_recall)
        result.update(eval_results_ndcg)
        result.update(eval_results_mrr)

    return result



def read_dataset(path, dataset_name):
    # need dataset name since each dataset has different format
    if dataset_name == "nq320k":
        with open(path) as f:
            data  = json.load(f)[:]
            all_labels = [str(item[1]) for item in data]
            queries = [item[0] for item in data]
        return queries, all_labels
    
    elif dataset_name == "scirepeval_search":
        ds = load_dataset("allenai/scirepeval", "search")

        ds_evaluation = ds["validation"]
        
        number_of_queries_with_no_rel_docs = 0
        queries = []
        all_labels = []
        for line in ds_evaluation:
            line_query = line.get("query")
            candidates = line.get("candidates")

            lowered_line_query = line_query.lower()
            candidates_titles = [c.get("title").lower() for c in candidates]
            if any([lowered_line_query in ctitle for ctitle in candidates_titles]): continue
            line_labels = [str(item["doc_id"]) for item in candidates if item["score"] > 0]
            if not line_labels: number_of_queries_with_no_rel_docs += 1

            queries.append(line_query)
            all_labels.append(line_labels)

        print("Number of datapoints", len(queries))
        print("Number of queries with no relevant docs", number_of_queries_with_no_rel_docs)
        return queries, all_labels
    
    elif dataset_name == "scidocs":
        # first process the qrels
        ds_qrels = load_dataset("BeIR/scidocs-qrels")

        qid2docids = {}
        for line in ds_qrels["test"]:
            qid = line["query-id"]
            docid = line["corpus-id"]
            score = line["score"]

            if score < 1: continue

            if qid not in qid2docids: qid2docids[qid] = []
            qid2docids[qid].append(docid)

        # then process the queries
        queries = []
        all_labels = []
        ds_queries = load_dataset("BeIR/scidocs", "queries")
        for line in ds_queries["queries"]:
            qid = line["_id"]

            line_query = line["text"].lower()
            line_labels = qid2docids[qid]

            queries.append(line_query)
            all_labels.append(line_labels)

        print("Number of datapoints", len(queries))
        # print("Number of queries with no relevant docs", number_of_queries_with_no_rel_docs)
        return queries, all_labels

    elif dataset_name == "scifact":
        # first process the qrels
        ds_qrels = load_dataset("BeIR/scifact-qrels")

        qid2docids = {}
        for split in ["train", "test"]:
            for line in ds_qrels[split]:
                qid = str(line["query-id"])
                docid = str(line["corpus-id"]).replace(",", "")
                score = line["score"]

                if score < 1: continue

                if qid not in qid2docids: qid2docids[qid] = []
                qid2docids[qid].append(docid)

        # then process the queries
        queries = []
        all_labels = []
        ds_queries = load_dataset("BeIR/scifact", "queries")
        for line in ds_queries["queries"]:
            qid = str(line["_id"])

            line_query = line["text"].lower()
            line_labels = qid2docids[qid]

            queries.append(line_query)
            all_labels.append(line_labels)

        print("Number of datapoints", len(queries))
        # print("Number of queries with no relevant docs", number_of_queries_with_no_rel_docs)
        return queries, all_labels


if __name__ == "__main__":
    # with open("/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/dev.json") as f:
    #     data  = json.load(f)[:]
    #     all_labels = [int(item[1]) for item in data]
    #     queries = [item[0] for item in data]




    queries, all_labels = read_dataset(
        path = GROUNDTRUTH_DATA_PATH, #"/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/dev.json",
        dataset_name= DATASET_NAME #"nq320k"
    )



    experiment_results = do_search_and_evaluate(queries = queries, all_labels=all_labels, show_progress_bar=True)
    experiment_results["name"] = EXPERIMENT_NAME
    experiment_results["when"] = get_current_date_string()


    with open("bm25_eval_results.txt", "a") as f:
        f.write(json.dumps(experiment_results))
        f.write("\n")