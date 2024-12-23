import os, json, logging, pytrec_eval
from pyserini.search.lucene import LuceneSearcher
# from pyserini.search import SimpleSearcher
from tqdm import tqdm
from evaluation.pytrec_eval_scripts import evaluation
from datetime import datetime
from datasets import load_dataset
from typing import List, Dict, Tuple


logger = logging.getLogger(__name__)

INDEX_PATH = os.environ["INDEX_PATH"]
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")

GROUNDTRUTH_DATA_PATH = os.getenv("GROUNDTRUTH_DATA_PATH")
DATASET_NAME = os.getenv("DATASET_NAME")

QUERY_EXPANSION_PATH = os.getenv("QUERY_EXPANSION_PATH", "")
EXPANSION_ONLY_PRESENT_KEYPHRASES = int(os.getenv("EXPANSION_ONLY_PRESENT_KEYPHRASES", 0))
NUMBER_OF_KEYWORDS_EACH_TYPE = int(os.getenv("NUMBER_OF_KEYWORDS_EACH_TYPE", 5))

SEARCHER = LuceneSearcher(INDEX_PATH)

USE_RM3 = os.getenv("USE_RM3", 0)


# METADATA_STORE_FOR_DEBUGGING = "/scratch/lamdo/eru_kg_retrieval_eval"


if QUERY_EXPANSION_PATH and os.path.exists(QUERY_EXPANSION_PATH):
    with open(QUERY_EXPANSION_PATH) as f:
        QUERY_EXPANSION = json.load(f)
    print(QUERY_EXPANSION[0])
else:
    QUERY_EXPANSION = None

def evaluate(qrels: Dict[str, Dict[str, int]], 
                results: Dict[str, Dict[str, float]], 
                k_values: List[int],
                ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    
    if ignore_identical_ids:
        logger.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
        popped = []
        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)
                    popped.append(pid)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
    
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    # with open(os.path.join(METADATA_STORE_FOR_DEBUGGING, f"{EXPERIMENT_NAME}.json"), "w") as f:
    #     json.dump({"scores": scores, "results": results, "qrels": qrels}, f)
    # print(len(scores))
    
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
    
    for eval in [ndcg, _map, recall, precision]:
        logger.info("\n")
        for k in eval.keys():
            logger.info("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision

def hit_template(hits):
    results = {}
    
    for qid, hit in hits.items():
        results[qid] = {}
        for i in range(0, len(hit)):
            results[qid][hit[i].docid] = hit[i].score
    
    return results

def get_current_date_string():
    current_date = datetime.now()
    return current_date.strftime("%d %B %Y")

def search_single_query(query, 
                        searcher = SEARCHER, 
                        top_k = 100, 
                        return_raw = False, 
                        bm25 = {"k1": 0.9, "b": 0.4}, 
                        fields={"contents": 1.0},
                        use_rm3 = USE_RM3):
    if not query: return []

    searcher.set_bm25(k1=bm25["k1"], b=bm25["b"])

    if use_rm3:
        searcher.set_rm3(10, 10, 0.5)

    hits = searcher.search(query, k = top_k, fields=fields, )

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
                           top_k = 1000,
                           show_progress_bar = False):
    
    all_search_results = search_multiple_queries(queries=queries, searcher=searcher, top_k=top_k, show_progress_bar=show_progress_bar)
    # print(all_labels[0], predictions[0])

    result = {}
    # for eval_top_k in [10, 100]:

    eval_top_k = top_k
    predictions = convert_to_pytrec_eval_format(queries = queries, all_search_results=[sr[:eval_top_k] for sr in all_search_results], type = "prediction")
    # evaluation_result = evaluation(qrels = qrels, predictions = predictions)
    evaluation_result = evaluate(qrels = qrels, results = predictions, k_values = [10, 100, 1000])

    result[eval_top_k] = evaluation_result

    return result


def do_query_expansion_using_keyphrases(query, expansion, 
                                        num_keyword_each_type = NUMBER_OF_KEYWORDS_EACH_TYPE,
                                        expansion_only_present_keyword=EXPANSION_ONLY_PRESENT_KEYPHRASES):
    present_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("present_keyphrases", [])[:num_keyword_each_type]
    absent_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("absent_keyphrases", [])[:num_keyword_each_type]

    if expansion_only_present_keyword:
        query = query + " " + " ".join(present_keyphrases)
    else:
        query = query + " " + " ".join(present_keyphrases) + " " + " ".join(absent_keyphrases)

    return query



def read_dataset(path, dataset_name, query_expansion = None):
    # need dataset name since each dataset has different format
    if dataset_name == "nq320k":
        with open(path) as f:
            data  = json.load(f)[:]
            all_labels = [str(item[1]) for item in data]
            queries = [item[0] for item in data]
        return queries, all_labels
    
    elif dataset_name == "scirepeval_search" or dataset_name == "scirepeval_search_evaluation":
        ds = load_dataset("allenai/scirepeval", "search")

        ds_evaluation = ds["evaluation"]

        if query_expansion is not None: assert len(query_expansion) == len(ds_evaluation)
        else: query_expansion = [None] * len(ds_evaluation)


        queries = []
        all_labels = []
        for line, expansion in zip(ds_evaluation, query_expansion):
            line_query = line.get("query")
            candidates = line.get("candidates")

            lowered_line_query = line_query.lower()
            candidates_titles = [c.get("title").lower() for c in candidates]

            # this is to remove easy cases where the query is the title of some paper
            # can comment this if needed
            # if any([lowered_line_query in ctitle for ctitle in candidates_titles]): continue


            # line_labels = [str(item["doc_id"]) for item in candidates if item["score"] > 0]
            line_labels = [{"docid": str(item["doc_id"]), "score": item["score"]} for item in candidates]
            # if not line_labels: number_of_queries_with_no_rel_docs += 1

            if expansion:
                # present_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("present_keyphrases", [])[:5]
                # absent_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("absent_keyphrases", [])[:5]
                # line_query = line_query + " " + " ".join(present_keyphrases) + " " + " ".join(absent_keyphrases)
                line_query = do_query_expansion_using_keyphrases(line_query, expansion)

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
        if query_expansion is not None: assert len(query_expansion) == len(ds_queries["queries"])
        else: query_expansion = [None] * len(ds_queries["queries"])
        for line, expansion in zip(ds_queries["queries"], query_expansion):
            qid = line["_id"]

            line_query = line["text"].lower()
            line_labels = qid2docids[qid]

            if expansion:
                # present_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("present_keyphrases", [])[:5]
                # absent_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("absent_keyphrases", [])[:5]
                # line_query = line_query + " " + " ".join(present_keyphrases) + " " + " ".join(absent_keyphrases)

                line_query = do_query_expansion_using_keyphrases(line_query, expansion)

                # tokens = line_query.split(" ")
                # line_query = " ".join(list(set(tokens)))

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
        if query_expansion is not None: assert len(query_expansion) == len(ds_queries["queries"])
        else: query_expansion = [None] * len(ds_queries["queries"])
        for line, expansion in zip(ds_queries["queries"], query_expansion):
            qid = str(line["_id"])

            line_query = line["text"].lower()
            line_labels = qid2docids[qid]
            
            if expansion:
                # present_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("present_keyphrases", [])[:5]
                # absent_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("absent_keyphrases", [])[:5]
                # line_query = line_query + " " + " ".join(present_keyphrases) + " " + " ".join(absent_keyphrases)

                line_query = do_query_expansion_using_keyphrases(line_query, expansion)

                # tokens = line_query.split(" ")
                # line_query = " ".join(list(set(tokens)))

            queries.append(line_query.strip())
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

        if query_expansion is not None: assert len(query_expansion) == len(ds_queries["queries"])
        else: query_expansion = [None] * len(ds_queries["queries"])
        for line, expansion in zip(ds_queries["queries"], query_expansion):
            qid = str(line["_id"])

            line_query = line["text"].lower()
            line_labels = qid2docids[qid]

            if expansion:
                # present_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("present_keyphrases", [])[:5]
                # absent_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("absent_keyphrases", [])[:5]
                # line_query = line_query + " " + " ".join(present_keyphrases) + " " + " ".join(absent_keyphrases)

                line_query = do_query_expansion_using_keyphrases(line_query, expansion)

                # tokens = line_query.split(" ")
                # line_query = " ".join(list(set(tokens)))

            queries.append(line_query)
            all_labels.append(line_labels)

        qrels = convert_to_pytrec_eval_format(queries = queries, all_search_results=all_labels)

        print("Number of datapoints", len(queries))
        # print("Number of queries with no relevant docs", number_of_queries_with_no_rel_docs)
        return queries, qrels
    

    elif dataset_name == "nfcorpus":
        # first process the qrels
        ds_qrels = load_dataset("BeIR/nfcorpus-qrels")

        qid2docids = {}
        for split in ["train", "validation", "test"]:
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
        ds_queries = load_dataset("BeIR/nfcorpus", "queries")

        if query_expansion is not None: assert len(query_expansion) == len(ds_queries["queries"])
        else: query_expansion = [None] * len(ds_queries["queries"])
        for line, expansion in zip(ds_queries["queries"], query_expansion):
            qid = str(line["_id"])

            line_query = line["text"].lower()
            line_labels = qid2docids[qid]

            if expansion:
                # present_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("present_keyphrases", [])[:5]
                # absent_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("absent_keyphrases", [])[:5]
                # line_query = line_query + " " + " ".join(present_keyphrases) + " " + " ".join(absent_keyphrases)

                line_query = do_query_expansion_using_keyphrases(line_query, expansion)

                # tokens = line_query.split(" ")
                # line_query = " ".join(list(set(tokens)))

            queries.append(line_query)
            all_labels.append(line_labels)

        qrels = convert_to_pytrec_eval_format(queries = queries, all_search_results=all_labels)

        print("Number of datapoints", len(queries))
        # print("Number of queries with no relevant docs", number_of_queries_with_no_rel_docs)
        return queries, qrels
    
    elif dataset_name == "doris_mae":
        from doris_mae_helper import compute_all_gpt_score
        # first process the qrels

        with open("/scratch/lamdo/doris-mae/DORIS-MAE_dataset_v1.json") as f:
            ds = json.load(f)
        
        all_gpt_score = compute_all_gpt_score(ds)

        qid2docids = {}
        for i, line in enumerate(all_gpt_score):
            qid = str(i)

            if qid not in qid2docids: qid2docids[qid] = []
            
            for docid, scores in line.items():
                if not scores[1] > 0: continue
                qid2docids[qid].append({
                    "docid": docid,
                    "score": scores[1]
                })

        # then process the queries
        queries = []
        all_labels = []

        if query_expansion is not None: assert len(query_expansion) == len(ds["Query"])
        else: query_expansion = [None] * len(ds["Query"])


        for i, (line, expansion) in enumerate(zip(ds["Query"], query_expansion)):
            qid = str(i)

            line_query = line["query_text"].lower()
            line_labels = qid2docids[qid]

            if expansion:
                # present_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("present_keyphrases", [])[:5]
                # absent_keyphrases = expansion.get("automatically_extracted_keyphrases", {}).get("absent_keyphrases", [])[:5]
                # line_query = line_query + " " + " ".join(present_keyphrases) + " " + " ".join(absent_keyphrases)

                line_query = do_query_expansion_using_keyphrases(line_query, expansion)

                # tokens = line_query.split(" ")
                # line_query = " ".join(list(set(tokens)))

            queries.append(line_query)
            all_labels.append(line_labels)

        qrels = convert_to_pytrec_eval_format(queries = queries, all_search_results=all_labels)

        print("Number of datapoints", len(queries))
        # print("Number of queries with no relevant docs", number_of_queries_with_no_rel_docs)
        return queries, qrels

        
    else: raise NotImplementedError


if __name__ == "__main__":
    # with open("/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/dev.json") as f:
    #     data  = json.load(f)[:]
    #     all_labels = [int(item[1]) for item in data]
    #     queries = [item[0] for item in data]




    queries, qrels = read_dataset(
        path = GROUNDTRUTH_DATA_PATH, #"/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/dev.json",
        dataset_name= DATASET_NAME, #"nq320k"
        query_expansion=QUERY_EXPANSION
    )


    print(queries[0])



    experiment_results = do_search_and_evaluate(queries = queries, qrels = qrels, show_progress_bar=True)
    experiment_results["name"] = EXPERIMENT_NAME
    experiment_results["when"] = get_current_date_string()
    # experiment_results["dataset_name"] = DATASET_NAME
    # experiment_results["model_name"] = EXPERIMENT_NAME.replace(DATASET_NAME + "_", "").replace("keyphrase_expansion_", "")


    with open("bm25_eval_results.txt", "a") as f:
        f.write(json.dumps(experiment_results))
        f.write("\n")


    # test = search_multiple_queries(queries = ["test search"], top_k = 10)
    # print(test)