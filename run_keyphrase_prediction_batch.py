import json, os
from datasets import load_dataset
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from typing import List

from src.two_stage_keyphrase_extraction_with_splade import keyphrase_extraction as splade_based_keyphrase_extraction
from src.embedrank_keyphrase_extraction import embedrank_keyphrase_extraction, embed_sentences_sentence_transformer, embed_sentences_sent2vec
from src.multipartiterank import keyphrase_extraction as multipartiterank_keyphrase_extraction
from src.process_dataset import process_dataset
from src.retrieval_based_ukg import keyphrase_generation_batch as retrieval_based_ukg_keyphrase_generation


RETRIEVAL_DATASETS = ["nq320k", "scirepeval_search", "scifact", "scidocs", 
                      "trec_covid", "scirepeval_search_validation_evaluation",
                      "scifact_queries"]

RESULTS_FOLDER = os.environ["RESULTS_FOLDER"]

MODEL_TO_USE = os.environ["MODEL_TO_USE"]
DATASET_TO_USE = os.environ["DATASET_TO_USE"]

PRECOMPUTED_REPRESENTATIONS_PATH = os.getenv("PRECOMPUTED_REPRESENTATIONS_PATH")

if PRECOMPUTED_REPRESENTATIONS_PATH and os.path.exists(PRECOMPUTED_REPRESENTATIONS_PATH):
    precomputed_representations = []
    with open(PRECOMPUTED_REPRESENTATIONS_PATH) as f:
        for line in f:
            rep = json.loads(line)

            precomputed_representations.append(rep)
    print(f"loaded precomputed representations from {PRECOMPUTED_REPRESENTATIONS_PATH}")
else:
    precomputed_representations = None


TOP_KS_TO_EVAL= [3,5,10]

RESULT_FILE = os.path.join(RESULTS_FOLDER, f"{DATASET_TO_USE}--{MODEL_TO_USE}.json")

def do_keyphrase_extraction_batch(docs: list, precomputed_representations: List[dict], top_k: 10):
    if MODEL_TO_USE == "embedrank_sent2vec":
        return [embedrank_keyphrase_extraction(doc, embed_func=embed_sentences_sent2vec, top_k = top_k) for doc in docs]
    
    elif MODEL_TO_USE == "uke_custom_trained_combined_references_v6_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                        informativeness_model_name="custom_trained_combined_references_v6",
                                                        apply_position_penalty=True, length_penalty=-0.25, alpha = 1.0,
                                                        precomputed_tokens_scores=precomputed_representations)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                        informativeness_model_name="custom_trained_combined_references_v6",
                                                        apply_position_penalty=True, length_penalty=-0.25, alpha = 0.8,
                                                        precomputed_tokens_scores=precomputed_representations)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_v6-2_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                        informativeness_model_name="custom_trained_combined_references_v6-2",
                                                        apply_position_penalty=True, length_penalty=-0.25, alpha = 0.8,
                                                        precomputed_tokens_scores=precomputed_representations)

    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v11-2_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                 informativeness_model_name="custom_trained_combined_references_v11-2",
                                                 apply_position_penalty=True, length_penalty=-0.25, alpha = 1.0,
                                                 precomputed_tokens_scores=precomputed_representations)
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v6-2_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                 informativeness_model_name="custom_trained_combined_references_v6-2",
                                                 apply_position_penalty=True, length_penalty=-0.25, alpha = 1.0,
                                                 precomputed_tokens_scores=precomputed_representations)
    
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                 informativeness_model_name="custom_trained_combined_references_nounphrase_v6-1",
                                                 apply_position_penalty=True, length_penalty=-0.25, alpha = 0.8,
                                                 precomputed_tokens_scores=precomputed_representations)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-2_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                 informativeness_model_name="custom_trained_combined_references_nounphrase_v6-2",
                                                 apply_position_penalty=True, length_penalty=-0.25, alpha = 0.8,
                                                 precomputed_tokens_scores=precomputed_representations)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-3_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                 informativeness_model_name="custom_trained_combined_references_nounphrase_v6-3",
                                                 apply_position_penalty=True, length_penalty=-0.25, alpha = 0.8,
                                                 precomputed_tokens_scores=precomputed_representations)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-4_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                 informativeness_model_name="custom_trained_combined_references_nounphrase_v6-4",
                                                 apply_position_penalty=True, length_penalty=-0.25, alpha = 0.8,
                                                 precomputed_tokens_scores=precomputed_representations)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-5_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                 informativeness_model_name="custom_trained_combined_references_nounphrase_v6-5",
                                                 apply_position_penalty=True, length_penalty=-0.25, alpha = 0.8,
                                                 precomputed_tokens_scores=precomputed_representations)
    

    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-1_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                 informativeness_model_name="custom_trained_combined_references_nounphrase_v7-1",
                                                 apply_position_penalty=True, length_penalty=-0.25, alpha = 0.8,
                                                 precomputed_tokens_scores=precomputed_representations)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-2_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                 informativeness_model_name="custom_trained_combined_references_nounphrase_v7-2",
                                                 apply_position_penalty=True, length_penalty=-0.25, alpha = 0.8,
                                                 precomputed_tokens_scores=precomputed_representations)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-3_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                 informativeness_model_name="custom_trained_combined_references_nounphrase_v7-3",
                                                 apply_position_penalty=True, length_penalty=-0.25, alpha = 0.8,
                                                 precomputed_tokens_scores=precomputed_representations)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-4_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                 informativeness_model_name="custom_trained_combined_references_nounphrase_v7-4",
                                                 apply_position_penalty=True, length_penalty=-0.25, alpha = 0.8,
                                                 precomputed_tokens_scores=precomputed_representations)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-5_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(docs, top_k = top_k, 
                                                 informativeness_model_name="custom_trained_combined_references_nounphrase_v7-5",
                                                 apply_position_penalty=True, length_penalty=-0.25, alpha = 0.8,
                                                 precomputed_tokens_scores=precomputed_representations)
    else:
        raise NotImplementedError


# get entire dataset
dataset = process_dataset(dataset_name=DATASET_TO_USE)


all_texts = [sample.get("text") for sample in dataset]
keyphrase_extraction_results_for_dataset = []

BATCH_SIZE = 20
for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc = "Extracting keyphrases"):
    batch = all_texts[i:i+BATCH_SIZE]
    if precomputed_representations:
        precomputed_representations_batch = precomputed_representations[i:i+BATCH_SIZE]
    else: precomputed_representations_batch = None

    batch_automatically_extracted_keyphrases = do_keyphrase_extraction_batch(
        batch,
        precomputed_representations=precomputed_representations_batch, 
        top_k = 50)
    keyphrase_extraction_results_for_dataset.extend(batch_automatically_extracted_keyphrases)



processed_dataset = []
for i, sample in tqdm(enumerate(dataset)):
    document = sample.get("text_not_lowered")
    present_keyphrases = sample.get("present_keyphrases")
    absent_keyphrases = sample.get("absent_keyphrases")

    automatically_extracted_keyphrases = keyphrase_extraction_results_for_dataset[i]
    automatically_extracted_keyphrases = {
        "present_keyphrases":  [item[0] for item in automatically_extracted_keyphrases["present"] if item[1] > 0],
        "absent_keyphrases":  [item[0] for item in automatically_extracted_keyphrases["absent"] if item[1] > 0],
    }

    line = {
        "document": document,
        "present_keyphrases": present_keyphrases,
        "absent_keyphrases": absent_keyphrases,
        "automatically_extracted_keyphrases": automatically_extracted_keyphrases,
    }

    # if DATASET_TO_USE in RETRIEVAL_DATASETS:
    line.pop("document", None)


    processed_dataset.append(line)

with open(RESULT_FILE, "w") as f:
    json.dump(processed_dataset, f, indent = 4)