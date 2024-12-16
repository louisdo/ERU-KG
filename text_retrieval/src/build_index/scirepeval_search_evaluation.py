import json, os, zipfile, requests, logging
from tqdm import tqdm
from datasets import load_dataset
from beir_generic_data_loader import GenericDataLoader
from utils import maybe_create_folder, remove_folder, convert_dataset_to_pyserini_format


logger = logging.getLogger(__name__)

INPUT_FILE = os.environ["INPUT_FILE"]
OUTPUT_FOLDER = os.environ["OUTPUT_FOLDER"] # "/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/pyserini_formatted_collection"
INDEX_FOLDER = os.getenv("INDEX_FOLDER")

KEYWORD_FOR_DOCUMENT_EXPANSION = os.getenv("KEYWORD_FOR_DOCUMENT_EXPANSION")

remove_folder(INDEX_FOLDER)
remove_folder(OUTPUT_FOLDER)
maybe_create_folder(OUTPUT_FOLDER)


# read scidocs dataset
ds_corpus = load_dataset("BeIR/trec-covid", "corpus")


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

documents = convert_dataset_to_pyserini_format(dataset = corpus, ids = ids, 
                                               keyword_for_document_expansion_path = KEYWORD_FOR_DOCUMENT_EXPANSION,
                                               convert_nounphrases_to_question=False)


output_folder = OUTPUT_FOLDER

for document in tqdm(documents[:]):
    document_id = document.get("id")
    with open(os.path.join(output_folder, f"{document_id}.json"), "w") as f:
        json.dump(document, f)


print(document)

command = f"""python -m pyserini.index.lucene --collection JsonCollection --input "{OUTPUT_FOLDER}" --index "{INDEX_FOLDER}" --generator DefaultLuceneDocumentGenerator --threads 8 --storePositions --storeDocvectors --storeRaw"""

os.system(command)

remove_folder(OUTPUT_FOLDER)