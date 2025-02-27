
import os, json, sys
from evaluation.evaluator import SupervisedEvaluator, SupervisedTask
from evaluation.encoders import Model


EMBEDDING_FILE = os.environ["EMBEDDING_FILE"]
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
OUTPUT_FILE = os.environ["OUTPUT_FILE"]


model = Model(variant="default", base_checkpoint="allenai/specter2_base")

# subtype = SupervisedTask.CLASSIFICATION
subtype = SupervisedTask.MULTILABEL_CLASSIFICATION
evaluator = SupervisedEvaluator("arxiv", subtype, 
                                model=model, 
                                meta_dataset=("/scratch/lamdo/arxiv_classification/arxiv_classification_20k", "default"), 
                                test_dataset=("/scratch/lamdo/arxiv_classification/arxiv_classification_20k"), 
                                metrics= ["f1_macro"],
                                batch_size=32)


results = evaluator.evaluate(EMBEDDING_FILE)
results["experiment_name"] = EXPERIMENT_NAME
with open(OUTPUT_FILE, "a+") as f:
    json.dump(results, f)
    f.write("\n")