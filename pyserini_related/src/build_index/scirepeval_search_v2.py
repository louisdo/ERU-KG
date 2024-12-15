import json, os
from tqdm import tqdm
from utils import maybe_create_folder, remove_folder, convert_dataset_to_pyserini_format_kg_index
from transformers import AutoTokenizer
from functools import lru_cache


INPUT_FILE = os.environ["INPUT_FILE"] # "/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/corpus_lite.json" 
OUTPUT_FOLDER = os.environ["OUTPUT_FOLDER"] # "/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/pyserini_formatted_collection"

KEYWORD_FOR_DOCUMENT_EXPANSION = os.getenv("KEYWORD_FOR_DOCUMENT_EXPANSION")
DOCUMENT_VECTORS_PATH = os.getenv("DOCUMENT_VECTORS_PATH")


INDEX_FOLDER = os.getenv("INDEX_FOLDER")

maybe_create_folder(OUTPUT_FOLDER)

TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")


@lru_cache(maxsize = 10000)
def tokenize(cand):
    return TOKENIZER.convert_ids_to_tokens(TOKENIZER(cand)["input_ids"][1:-1])

if DOCUMENT_VECTORS_PATH is not None:
    document_vectors = []
    with open(DOCUMENT_VECTORS_PATH) as f:
        for line in f:
            dvec = json.loads(line)
            document_vectors.append(dvec)

else: document_vectors = None

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

documents = convert_dataset_to_pyserini_format_kg_index(
    dataset = corpus, 
    ids = ids, 
    documents_vectors = document_vectors,
    keyword_for_document_expansion_path = KEYWORD_FOR_DOCUMENT_EXPANSION,
    length_penalty=-0.25,
    tokenizer=tokenize)


output_folder = OUTPUT_FOLDER
print(documents[0])
for document in tqdm(documents[:]):
    document_id = document.get("id")
    with open(os.path.join(output_folder, f"{document_id}.json"), "w") as f:
        json.dump(document, f)


command = f"""python -m pyserini.index.lucene --collection JsonCollection --input "{OUTPUT_FOLDER}" --index "{INDEX_FOLDER}" --generator DefaultLuceneDocumentGenerator --threads 8 --storePositions --storeDocvectors --storeRaw"""

os.system(command)

remove_folder(OUTPUT_FOLDER)