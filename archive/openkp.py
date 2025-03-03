import json, os
from datasets import load_dataset
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer

from erukg.two_stage_keyphrase_extraction_with_splade import keyphrase_extraction as splade_based_keyphrase_extraction
from erukg.embedrank_keyphrase_extraction import embedrank_keyphrase_extraction, embed_sentences_sentence_transformer, embed_sentences_sent2vec


RESULTS_FOLDER = os.environ["RESULTS_FOLDER"]

MODEL_TO_USE = os.environ["MODEL_TO_USE"]
TOP_K = int(os.environ["TOP_K"])

RESULT_FILE = os.path.join(RESULTS_FOLDER, f"openkp--{MODEL_TO_USE}--{TOP_K}.json")


def do_keyphrase_extraction(doc, top_k = 10):
    if MODEL_TO_USE == "embedrank_sent2vec":
        return embedrank_keyphrase_extraction(doc, embed_func=embed_sentences_sent2vec, top_k = top_k)
    
    elif MODEL_TO_USE == "embedrank_sentence_transformers":
        return embedrank_keyphrase_extraction(doc, embed_func=embed_sentences_sentence_transformer, top_k = top_k)
    
    elif MODEL_TO_USE == "splade_based":
        return splade_based_keyphrase_extraction(doc, top_k = top_k)
    
    else:
        raise NotImplementedError

# get entire dataset
dataset = load_dataset("midas/openkp", "raw", trust_remote_code=True)

STEMMER = PorterStemmer()


def precision_recall_f1(prediction: list, groundtruth: list):
    # convert to sets
    if not prediction or not groundtruth: return 0, 0 ,0
    set_prediction = set([STEMMER.stem(item) for item in prediction])
    set_groundtruth = set([STEMMER.stem(item) for item in groundtruth])

    prediction_groundtruth_intersection = set_groundtruth.intersection(set_prediction)

    precision = len(prediction_groundtruth_intersection) / len(set_prediction)
    recall = len(prediction_groundtruth_intersection) / len(set_groundtruth)

    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0

    return precision, recall, f1


processed_dataset = []
for sample in tqdm(dataset["test"]):
    document = " ".join(sample["document"])
    present_keyphrases = sample["extractive_keyphrases"]
    absent_keyphrases = sample["abstractive_keyphrases"]

    if isinstance(document, str):
        automatically_extracted_keyphrases = do_keyphrase_extraction(document, top_k = TOP_K)
        automatically_extracted_keyphrases = [item[0] for item in automatically_extracted_keyphrases]
    else: 
        print(type(document))
        automatically_extracted_keyphrases = []

    p,r,f = precision_recall_f1(prediction = automatically_extracted_keyphrases, groundtruth = present_keyphrases)


    line = {
        "document": document,
        "present_keyphrases": present_keyphrases,
        "absent_keyphrases": absent_keyphrases,
        "automatically_extracted_keyphrases": automatically_extracted_keyphrases,
        "precision": p,
        "recall": r,
        "f1": f
    }
    processed_dataset.append(line)

with open(RESULT_FILE, "w") as f:
    json.dump(processed_dataset, f, indent = 4)