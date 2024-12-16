import json, random
from datasets import load_dataset
from tqdm import tqdm


OUTPUT_FILE = "/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search_evaluation.jsonl"

ds = load_dataset("allenai/scirepeval", "search")


def get_all_documents_from_scirepeval_search_dataset(ds, filename):
    splits = ["evaluation"] #, "train"][:2]

    count = 0

    visited = set([])
    with open(filename, "w") as outfile:
        for split in splits:
            ds_split = ds[split]

            for line in tqdm(ds_split):
                candidates = line.get("candidates", [])
                if not candidates: continue

                for candidate in candidates:
                    if candidate["doc_id"] in visited: continue
                    visited.add(candidate["doc_id"])

                    # write to jsonl file
                    json.dump(candidate, outfile)
                    outfile.write("\n")

                    count += 1
    print(count)

get_all_documents_from_scirepeval_search_dataset(ds, OUTPUT_FILE)