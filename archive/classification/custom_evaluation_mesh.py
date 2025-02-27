import os, json
from evaluation.evaluator import SupervisedEvaluator, SupervisedTask
from evaluation.encoders import Model


EMBEDDING_FILE = os.environ["EMBEDDING_FILE"]
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]

model = Model(variant="default", base_checkpoint="allenai/specter2_base")
subtype = SupervisedTask.CLASSIFICATION

evaluator = SupervisedEvaluator("MeSH", subtype, 
                                model=model, 
                                meta_dataset=("allenai/scirepeval", "mesh_descriptors"), 
                                test_dataset=("allenai/scirepeval_test", "mesh_descriptors"), 
                                metrics= ["f1_macro"])


results = evaluator.evaluate(EMBEDDING_FILE)
results["experiment_name"] = EXPERIMENT_NAME
with open(f"experiments/mesh.json", "a+") as f:
    json.dump(results, f)