import json, os
from tqdm import tqdm
from src.process_dataset import process_dataset, process_dataset_with_doct5query
from src.nounphrase_extractor import CandidateExtractorRegExpNLTK

CANDEXT = CandidateExtractorRegExpNLTK([1,5])


DATASET_TO_USE = os.environ["DATASET_TO_USE"]
RESULTS_FOLDER = os.environ["RESULTS_FOLDER"]


RESULT_FILE = os.path.join(RESULTS_FOLDER, f"{DATASET_TO_USE}--doct5queries.json")

# get entire dataset
dataset = process_dataset(dataset_name=DATASET_TO_USE)

generated_queries = process_dataset_with_doct5query(dataset_name = DATASET_TO_USE)



processed_dataset = []
for sample in tqdm(dataset):
    # document = sample.get("text")
    doc_id = sample.get("doc_id")

    generated_queries_for_doc = generated_queries.get(doc_id)
    generated_queries_for_doc = generated_queries_for_doc if generated_queries_for_doc else []
    # nounphrases_from_generated_queries_for_doc = []
    # for gq in generated_queries_for_doc:
    #     gqnp = CANDEXT(gq)
    #     nounphrases_from_generated_queries_for_doc.extend(gqnp)

    line = {
        # "document": document,
        "automatically_extracted_keyphrases": generated_queries_for_doc,
    }


    processed_dataset.append(line)

with open(RESULT_FILE, "w") as f:
    json.dump(processed_dataset, f, indent = 4)