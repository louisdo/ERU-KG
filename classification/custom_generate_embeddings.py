import os
from evaluation.eval_datasets import SimpleDataset  
from evaluation.evaluator import Evaluator
from evaluation.encoders import Model



EMBEDDINGS_SAVE_PATH = os.environ["EMBEDDINGS_SAVE_PATH"] # "/scratch/lamdo/scirepeval_classification/embeddings/embeddingsspecter2_base_fos.json"
DATASET_PATH = os.environ["DATASET_PATH"]
KEYPHRASE_EXPANSION = int(os.getenv("KEYPHRASE_EXPANSION", "1"))

print(["title", "abstract", "keyphrase_expansion"] if KEYPHRASE_EXPANSION else ["title", "abstract"])

model = Model(variant="default", base_checkpoint="allenai/specter2_base")

model.task_id = "[CLF]"

dataset = DATASET_PATH#"/scratch/lamdo/scirepeval_classification/fos/scirepeval_fos_test_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json"

evaluator = Evaluator(
    name = "fos",
    meta_dataset=dataset, 
    dataset_class=SimpleDataset,
    model=model,
    batch_size=4,
    fields=["title", "abstract", "keyphrase_expansion"] if KEYPHRASE_EXPANSION else ["title", "abstract"],
    key="doc_id"
)
embeddings = evaluator.generate_embeddings(save_path=EMBEDDINGS_SAVE_PATH)

# {title} [SEP] {abstract} [SEP] {keyphrase_expansion}