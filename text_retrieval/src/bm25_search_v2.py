import os, json
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
from evaluation.pytrec_eval_scripts import evaluation
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


def convert_to_pytrec_eval_format(queries, all_search_results, type = "relevance"):
    """
    queries: [q1, q2, ...]
    all_search_results: [[{'docid': '22711954', 'score': 4.043900012969971}, ...]]
    """

    assert len(queries) == len(all_search_results)

    score_converter = {
        "relevance": lambda x: int(x),
        "prediction": lambda x:float(x)
    }

    res = {}
    for query, search_results in zip(queries, all_search_results):
        if query not in res:
            res[query] = {}
        

        for sr in search_results:
            docid = str(sr["docid"])
            score = score_converter[type](sr["score"])

            res[query][docid] = score

    return res





def do_search_and_evaluate(queries, qrels,
                           searcher = SEARCHER, 
                           top_k = 10,
                           show_progress_bar = False):
    
    all_search_results = search_multiple_queries(queries=queries, searcher=searcher, top_k=top_k, show_progress_bar=show_progress_bar)
    # print(all_labels[0], predictions[0])

    result = {}
    # for eval_top_k in [10, 100]:

    eval_top_k = top_k
    predictions = convert_to_pytrec_eval_format(queries = queries, all_search_results=[sr[:eval_top_k] for sr in all_search_results], type = "prediction")
    evaluation_result = evaluation(qrels = qrels, predictions = predictions)

    result[eval_top_k] = evaluation_result

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

        ds_evaluation = ds["evaluation"]


        queries = []
        all_labels = []
        for line in ds_evaluation:
            line_query = line.get("query")
            candidates = line.get("candidates")

            lowered_line_query = line_query.lower()
            candidates_titles = [c.get("title").lower() for c in candidates]

            # this is to remove easy cases where the query is the title of some paper
            # can comment this if needed
            if any([lowered_line_query in ctitle for ctitle in candidates_titles]): continue


            # line_labels = [str(item["doc_id"]) for item in candidates if item["score"] > 0]
            line_labels = [{"docid": str(item["doc_id"]), "score": item["score"]} for item in candidates]
            # if not line_labels: number_of_queries_with_no_rel_docs += 1

            queries.append(line_query)
            all_labels.append(line_labels)

        qrels = convert_to_pytrec_eval_format(queries = queries, all_search_results=all_labels)

        print("Number of datapoints", len(queries))
        return queries, qrels
    
    elif dataset_name == "scidocs":
        # first process the qrels
        ds_qrels = load_dataset("BeIR/scidocs-qrels")

        qid2docids = {}
        for line in ds_qrels["test"]:
            qid = line["query-id"]
            docid = line["corpus-id"]
            score = line["score"]

            # if score < 1: continue

            if qid not in qid2docids: qid2docids[qid] = []
            qid2docids[qid].append({
                "docid": docid,
                "score": score
            })

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

        qrels = convert_to_pytrec_eval_format(queries = queries, all_search_results=all_labels)

        print("Number of datapoints", len(queries))
        # print("Number of queries with no relevant docs", number_of_queries_with_no_rel_docs)
        return queries, qrels

    elif dataset_name == "scifact":
        # first process the qrels
        ds_qrels = load_dataset("BeIR/scifact-qrels")

        qid2docids = {}
        for split in ["train", "test"]:
            for line in ds_qrels[split]:
                qid = str(line["query-id"])
                docid = str(line["corpus-id"]).replace(",", "")
                score = line["score"]

                # if score < 1: continue

                if qid not in qid2docids: qid2docids[qid] = []
                qid2docids[qid].append({
                    "docid": docid,
                    "score": score
                })

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

        qrels = convert_to_pytrec_eval_format(queries = queries, all_search_results=all_labels)

        print("Number of datapoints", len(queries))
        # print("Number of queries with no relevant docs", number_of_queries_with_no_rel_docs)
        return queries, qrels
    
    elif dataset_name == "trec_covid":
        # first process the qrels
        ds_qrels = load_dataset("BeIR/trec-covid-qrels")

        qid2docids = {}
        for split in ["test"]:
            for line in ds_qrels[split]:
                qid = str(line["query-id"])
                docid = str(line["corpus-id"]).replace(",", "")
                score = line["score"]

                # if score < 1: continue

                if qid not in qid2docids: qid2docids[qid] = []
                qid2docids[qid].append({
                    "docid": docid,
                    "score": score
                })

        # then process the queries
        queries = []
        all_labels = []
        ds_queries = load_dataset("BeIR/trec-covid", "queries")
        for line in ds_queries["queries"]:
            qid = str(line["_id"])

            line_query = line["text"].lower()
            line_labels = qid2docids[qid]

            queries.append(line_query)
            all_labels.append(line_labels)

        qrels = convert_to_pytrec_eval_format(queries = queries, all_search_results=all_labels)

        print("Number of datapoints", len(queries))
        # print("Number of queries with no relevant docs", number_of_queries_with_no_rel_docs)
        return queries, qrels


if __name__ == "__main__":
    # with open("/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/dev.json") as f:
    #     data  = json.load(f)[:]
    #     all_labels = [int(item[1]) for item in data]
    #     queries = [item[0] for item in data]




    queries, qrels = read_dataset(
        path = GROUNDTRUTH_DATA_PATH, #"/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/dev.json",
        dataset_name= DATASET_NAME #"nq320k"
    )



    experiment_results = do_search_and_evaluate(queries = queries, qrels = qrels, show_progress_bar=True)
    experiment_results["name"] = EXPERIMENT_NAME
    experiment_results["when"] = get_current_date_string()


    with open("bm25_eval_results.txt", "a") as f:
        f.write(json.dumps(experiment_results))
        f.write("\n")


    # test = search_multiple_queries(queries = ["test search"], top_k = 10)
    # print(test)