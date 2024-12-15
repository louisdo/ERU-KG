import json
from datasets import load_dataset


def read_dataset(path, dataset_name):
    # need dataset name since each dataset has different format
    if dataset_name == "nq320k":
        with open(path) as f:
            data  = json.load(f)[:]
            all_labels = [int(item[1]) for item in data]
            queries = [item[0] for item in data]
        return queries, all_labels
    
    elif dataset_name == "scirepeval_search":
        ds = load_dataset("allenai/scirepeval", "search")

        ds_evaluation = ds["validation"]
        
        number_of_queries_with_no_rel_docs = 0
        number_of_impossible_queries = 0
        queries = []
        all_labels = []
        for line in ds_evaluation:
            line_query = line.get("query")
            candidates = line.get("candidates")

            lowered_line_query = line_query.lower()
            candidates_texts = [c.get("title", "").lower() + "\n" + c.get("abstract", "").lower() for c in candidates if c.get("score") > 0]
            
            lowered_line_query_words = lowered_line_query.split(" ")
            query_exist_in_candidates = [any([query_word in ctext for query_word in lowered_line_query_words]) for ctext in candidates_texts]
            if not any(query_exist_in_candidates): number_of_impossible_queries += 1

            line_labels = [int(item["doc_id"]) for item in candidates if item["score"] > 0]
            if not line_labels: number_of_queries_with_no_rel_docs += 1

            queries.append(line_query)
            all_labels.append(line_labels)

        print("Number of datapoints", len(queries))
        print("Number of queries with no relevant docs", number_of_queries_with_no_rel_docs)
        print("Number of impossible queries", number_of_impossible_queries)
        return queries, all_labels
    

if __name__ == "__main__":
    read_dataset(None, "scirepeval_search")