# EMBEDDING_FILE="/scratch/lamdo/scirepeval_classification/embeddings/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" python custom_evaluation_fos.py

import os, json, sys
sys.path.append('../scirepeval')
from evaluation.evaluator import SupervisedEvaluator, SupervisedTask
from evaluation.encoders import Model


EMBEDDING_FILE = os.environ["EMBEDDING_FILE"]
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]

model = Model(variant="default", base_checkpoint="allenai/specter2_base")
subtype = SupervisedTask.MULTILABEL_CLASSIFICATION
evaluator = SupervisedEvaluator("Fields of study", subtype, 
                                model=model, 
                                meta_dataset=("allenai/scirepeval", "fos"), 
                                test_dataset=("allenai/scirepeval_test", "fos"), 
                                metrics= ["f1_macro"])


results = evaluator.evaluate(EMBEDDING_FILE)
results["experiment_name"] = EXPERIMENT_NAME
with open(f"test_{EXPERIMENT_NAME}.json", "a+") as f:
    json.dump(results, f)