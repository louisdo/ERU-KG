import pandas as pd
import numpy as np


INPUT_FILE = "/home/lamdo/keyphrase_informativeness_test/view.csv"

df = pd.read_csv(INPUT_FILE)

# group models by dataset
model_performances_by_dataset = {}
# iterate over df as list of dicts
for row in df.to_dict(orient="records"):
    dataset = row["dataset_name"]
    if dataset not in model_performances_by_dataset:
        model_performances_by_dataset[dataset] = []
    model_performances_by_dataset[dataset].append(row)


metrics_to_get_ranking = ["f1@3", "f1@5", "f1@10"]

for dataset in model_performances_by_dataset:
    for metric in metrics_to_get_ranking:
        ranking_indices = np.argsort([-1 * row[metric] for row in model_performances_by_dataset[dataset]])
        for i, index in enumerate(ranking_indices):
            model_performances_by_dataset[dataset][index][f"rank_{metric}"] = i + 1


# convert to csv where each row is a model and columns are ranking of the model on each dataset


model_performances = {}
for dataset in model_performances_by_dataset:
    for row in model_performances_by_dataset[dataset]:
        model = row["model_name"]
        if model not in model_performances:
            model_performances[model] = {}

        for metric in metrics_to_get_ranking:
            # model_performances[model][f"rank_{metric}_{dataset}"] = row[f"rank_{metric}"]
            model_performances[model][f"{metric}_{dataset}"] = row[f"{metric}"]

res = []
for model in model_performances:
    line = model_performances[model]
    line["model_name"] = model
    res.append(line)
res = pd.DataFrame(res)
#make column model_name the index
res = res.set_index("model_name")
res.to_csv("view_rankings.csv")