import json
import pandas as pd

DATASETS = ["scidocs", "scifact", "nfcorpus", "trec_covid", "doris_mae"]

EXPERIMENT_TYPES = ["[query expansion]", "[doc expansion]", "[query + doc expansion]", "[query present keyphrase expansion]", "[doc present keyphrase expansion]"]
KEYPHRASE_GENERATION_MODELS_NAMES = {"ERU-KG", "AutoKeyGen", "UOKG", "CopyRNN"}

METRICS_TO_USE = ["Recall@1000"]


GROUPS_KG = {}

for experiment_type in EXPERIMENT_TYPES:
    GROUPS_KG[f"ERU-KG {experiment_type}"] = [f"retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-{i}_position_penalty+length_penalty {experiment_type}" for i in range(1, 4)]
    GROUPS_KG[f"AutoKeyGen {experiment_type}"] = [f"autokeygen-{i} {experiment_type}" for i in range(1, 4)]
    GROUPS_KG[f"UOKG {experiment_type}"] = [f"uokg-{i} {experiment_type}" for i in range(1, 4)]
    GROUPS_KG[f"CopyRNN {experiment_type}"] = [f"copyrnn-{i} {experiment_type}" for i in range(1, 4)]

GROUPS_TRADITIONAL_MODELS = {
    "DocT5query": ["doct5queries"] + [None] * (3 - 1),
    "DocT5query+RM3": ["doct5queries_rm3"] + [None] * (3 - 1),
    "RM3": ["rm3"] + [None] * (3 - 1),
}


GROUPS = {**GROUPS_KG, **GROUPS_TRADITIONAL_MODELS}

def get_group_name_and_run_index_from_experiment_name(name):
    for group in GROUPS:
        for i, run_name in enumerate(GROUPS[group]):
            if not run_name: continue
            if run_name ==name:
                return group, i
            
    return None, None

def get_dataset_name_from_experiment_name(name):
    for dataset in DATASETS:
        if dataset in name:
            return dataset
    return None

data = []
with open("bm25_eval_results_23december2024.txt") as f:
    for line in f:
        jline = json.loads(line)
        exp_name = jline["name"]

        dataset_name = get_dataset_name_from_experiment_name(exp_name)
        if not dataset_name: continue

        _model_name = exp_name.replace(dataset_name + "_", "").replace("keyphrase_expansion_", "")

        if _model_name == exp_name: 
            model_name = "no expansion"
            run_index = None
        else:
            model_name, run_index = get_group_name_and_run_index_from_experiment_name(_model_name)
            if not model_name: 
                print(_model_name)
                continue

        results = {"model_name": model_name, "run_index": run_index, "dataset_name": dataset_name}
        for item in jline["1000"]:
            results.update({k:v for k,v in item.items() if k in METRICS_TO_USE})


        data.append(results)


data = list(sorted(data, key = lambda x: [x["dataset_name"], x["model_name"], x["run_index"]]))
df = pd.DataFrame(data)

first_column = df.pop('dataset_name')
df.insert(0, 'dataset_name', first_column)

df.to_csv("bm25_eval_results_23december2024.csv", index = False)