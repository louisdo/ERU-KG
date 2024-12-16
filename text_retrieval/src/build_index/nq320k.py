import json, os
from tqdm import tqdm


INPUT_FILE = os.environ["INPUT_FILE"] # "/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/corpus_lite.json" 
OUTPUT_FOLDER = os.environ["OUTPUT_FOLDER"] # "/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/pyserini_formatted_collection"

KEYWORD_FOR_DOCUMENT_EXPANSION = os.getenv("KEYWORD_FOR_DOCUMENT_EXPANSION")


def convert_dataset_to_pyserini_format(dataset):
    # dataset is expected to be a list
    assert isinstance(dataset, list)

    if KEYWORD_FOR_DOCUMENT_EXPANSION:
        with open(KEYWORD_FOR_DOCUMENT_EXPANSION) as f:
            expansion_keywords = json.load(f)

        assert len(expansion_keywords) == len(dataset)
    else:
        expansion_keywords = [{} for _ in range(len(dataset))]

    documents = []
    for i, line in enumerate(dataset):
        kw_exp = expansion_keywords[i].get("automatically_extracted_keyphrases")
        kw_exp = kw_exp if kw_exp else []

        if kw_exp:
            kw_exp_string = ", ".join(kw_exp)
            kw_exp_string = f"\n\nKeywords: {kw_exp_string}"
        else: kw_exp_string = ""


        documents.append({"id": i, "contents": f"{line}{kw_exp_string}"})

    return documents

with open(INPUT_FILE) as f:
    corpus = json.load(f)

documents = convert_dataset_to_pyserini_format(dataset = corpus)


output_folder = OUTPUT_FOLDER

for document in tqdm(documents[:]):
    document_id = document.get("id")
    with open(os.path.join(output_folder, f"{document_id}.json"), "w") as f:
        json.dump(document, f)

