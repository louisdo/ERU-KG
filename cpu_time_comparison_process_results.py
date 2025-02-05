import json
import pandas as pd
from copy import deepcopy


with open("cpu_time_comparison.txt") as f:
    eval_data = []
    for line in f:
        eval_data.append(json.loads(line))



GROUPS = {}

GROUPS["ERU-KG-base"] = [f"retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-{i}_position_penalty+length_penalty" for i in range(1, 4)]
GROUPS["ERU-KG-small"] = [f"retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-{i}_position_penalty+length_penalty" for i in range(1, 4)]
GROUPS["AutoKeyGen"] = [f"autokeygen-{i}" for i in range(1, 4)]
GROUPS["UOKG"] = [f"uokg-{i}" for i in range(1, 4)]
GROUPS["CopyRNN"] = [f"copyrnn-{i}" for i in range(1, 4)]
GROUPS["TPG"] = [f"tpg-{i}" for i in range(1, 4)]
GROUPS["EmbedRank"] = ["embedrank_sent2vec"]
GROUPS["TextRank"] = ["textrank"]
GROUPS["MultiPartiteRank"] = ["multipartiterank"]
GROUPS["PromptRank"] = ["promptrank"]
GROUPS["ERU-KG-base [only extract]"] = [f"retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-{i}_position_penalty+length_penalty_alpha_1" for i in range(1, 4)]
GROUPS["ERU-KG-small [only extract]"] = [f"retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-{i}_position_penalty+length_penalty_alpha_1" for i in range(1, 4)]
GROUPS["PromptKP"] = ["promptkp"]


def get_group_name_and_run_index_from_experiment_name(name):
    for group in GROUPS:
        for i, run_name in enumerate(GROUPS[group]):
            if not run_name: continue
            if run_name ==name:
                return group, i
            
    return None, None

new_eval_data = []
for line in eval_data:
    newline = deepcopy(line)

    del newline["model"]

    model, _ = get_group_name_and_run_index_from_experiment_name(line["model"])
    
    newline["model"] = model

    new_eval_data.append(newline)

df = pd.DataFrame(new_eval_data)


first_column = df.pop('model')
df.insert(0, 'model', first_column)

second_column = df.pop('run_index')
df.insert(1, 'run_index', second_column)


df.to_csv("cpu_time_comparison.csv", index=False)