import json, os, zipfile, requests, logging
from tqdm import tqdm
from datasets import load_dataset
from beir_generic_data_loader import GenericDataLoader
from utils import maybe_create_folder, remove_folder, convert_dataset_to_pyserini_format


logger = logging.getLogger(__name__)


OUTPUT_FOLDER = os.environ["OUTPUT_FOLDER"] # "/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/pyserini_formatted_collection"
INDEX_FOLDER = os.getenv("INDEX_FOLDER")

KEYWORD_FOR_DOCUMENT_EXPANSION = os.getenv("KEYWORD_FOR_DOCUMENT_EXPANSION")
EXPANSION_ONLY_PRESENT_KEYPHRASES = int(os.getenv("EXPANSION_ONLY_PRESENT_KEYPHRASES", 0))
NUMBER_OF_KEYWORDS_EACH_TYPE = int(os.getenv("NUMBER_OF_KEYWORDS_EACH_TYPE", 10))

USE_KEYWORDS_INSTEAD_OF_DOCUMENT = int(os.getenv("USE_KEYWORDS_INSTEAD_OF_DOCUMENT", 0))

remove_folder(INDEX_FOLDER)
remove_folder(OUTPUT_FOLDER)
maybe_create_folder(OUTPUT_FOLDER)


data = []
with open("/scratch/lamdo/acm-cr/data/docs/collection.jsonl") as f:
    for line in f:
        jline = json.loads(line)
        data.append(jline)


corpus = []
ids = []
for line in tqdm(data):
    jline = line
    
    title = jline["title"]
    abstract = jline.get("abstract")

    title = title.replace("_", " ")
    abstract = abstract.replace("\n", " ")
    

    text = f"{title}. {abstract}"
    corpus.append(text)

    doc_id = jline["id"]
    ids.append(doc_id)


documents = convert_dataset_to_pyserini_format(dataset = corpus, ids = ids, 
                                               keyword_for_document_expansion_path = KEYWORD_FOR_DOCUMENT_EXPANSION,
                                               convert_nounphrases_to_question=False,
                                               num_keyword_each_type = NUMBER_OF_KEYWORDS_EACH_TYPE,
                                               expansion_only_present_keyword=EXPANSION_ONLY_PRESENT_KEYPHRASES,
                                               use_keywords_instead_of_document=USE_KEYWORDS_INSTEAD_OF_DOCUMENT)


output_folder = OUTPUT_FOLDER

for document in tqdm(documents[:]):
    document_id = document.get("id").replace("/", "_").replace(".", "_")
    with open(os.path.join(output_folder, f"{document_id}.json"), "w") as f:
        json.dump(document, f)


print(document)

command = f"""python -m pyserini.index.lucene --collection JsonCollection --input "{OUTPUT_FOLDER}" --index "{INDEX_FOLDER}" --generator DefaultLuceneDocumentGenerator --threads 8 --storePositions --storeDocvectors --storeRaw"""

os.system(command)

remove_folder(OUTPUT_FOLDER)