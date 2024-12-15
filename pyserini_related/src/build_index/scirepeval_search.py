import json, os
from tqdm import tqdm
from utils import maybe_create_folder, convert_dataset_to_pyserini_format


INPUT_FILE = os.environ["INPUT_FILE"] # "/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/corpus_lite.json" 
OUTPUT_FOLDER = os.environ["OUTPUT_FOLDER"] # "/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/pyserini_formatted_collection"

KEYWORD_FOR_DOCUMENT_EXPANSION = os.getenv("KEYWORD_FOR_DOCUMENT_EXPANSION")

maybe_create_folder(OUTPUT_FOLDER)

with open(INPUT_FILE) as f:
    corpus = []
    ids = []
    for line in tqdm(f):
        jline = json.loads(line)
        
        title = jline["title"]
        abstract = jline["abstract"]
        text = f"{title}\n{abstract}"
        corpus.append(text)

        doc_id = jline["doc_id"]
        ids.append(doc_id)

documents = convert_dataset_to_pyserini_format(dataset = corpus, 
                                               ids = ids, 
                                               keyword_for_document_expansion_path = KEYWORD_FOR_DOCUMENT_EXPANSION,
                                               apply_expansion_using_keyword=True)


output_folder = OUTPUT_FOLDER

for document in tqdm(documents[:]):
    document_id = document.get("id")
    with open(os.path.join(output_folder, f"{document_id}.json"), "w") as f:
        json.dump(document, f)

print(document)