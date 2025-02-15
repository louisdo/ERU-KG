import pandas as pd
from copy import deepcopy



GROUPS = {}

GROUPS["ERU-KG-base"] = [f"retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-{i}_position_penalty+length_penalty" for i in range(1, 4)]
GROUPS["ERU-KG-base (no queries)"] = [f"retrieval_based_ukg_custom_trained_combined_references_no_queries_nounphrase_v8-{i}_position_penalty+length_penalty" for i in range(1, 4)]
GROUPS["ERU-KG-base (no titles)"] = [f"retrieval_based_ukg_custom_trained_combined_references_no_titles_nounphrase_v8-{i}_position_penalty+length_penalty" for i in range(1, 4)]
GROUPS["ERU-KG-base (no cc)"] = [f"retrieval_based_ukg_custom_trained_combined_references_no_cc_nounphrase_v8-{i}_position_penalty+length_penalty" for i in range(1, 4)]
GROUPS["ERU-KG-base (neighbor size 50)"] = [f"retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-{i}_position_penalty+length_penalty_neighborsize_50" for i in range(1, 4)]
GROUPS["ERU-KG-base (neighbor size 10)"] = [f"retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-{i}_position_penalty+length_penalty_neighborsize_10" for i in range(1, 4)]

GROUPS["ERU-KG-small"] = [f"retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-{i}_position_penalty+length_penalty" for i in range(1, 4)]
GROUPS["AutoKeyGen"] = [f"autokeygen-{i}" for i in range(1, 4)]
GROUPS["UOKG"] = [f"uokg-{i}" for i in range(1, 4)]
GROUPS["CopyRNN"] = [f"copyrnn-{i}" for i in range(1, 4)]
GROUPS["TPG"] = [f"tpg-{i}" for i in range(1, 4)]
GROUPS["EmbedRank (SBERT)"] = ["embedrank_sentence_transformers_all-MiniLM-L12-v2"]
GROUPS["EmbedRank"] = ["embedrank_sent2vec"]
GROUPS["TextRank"] = ["textrank"]
GROUPS["MultiPartiteRank"] = ["multipartiterank"]
GROUPS["PromptRank"] = ["promptrank"]


def get_group_name_and_run_index_from_experiment_name(name):
    for group in GROUPS:
        for i, run_name in enumerate(GROUPS[group]):
            if not run_name: continue
            if run_name ==name:
                return group, i
            
    return None, None

def compute_average_performance(input_df):
    avg_df = input_df.drop("dataset_name", axis = 1).groupby(['model_name', 'run_index']).mean().reset_index()
    avg_df['dataset_name'] = 'average'
    avg_df = avg_df[input_df.columns]

    return avg_df

df = pd.read_csv("view_12feb2025.csv")

processed_dataset = []
for line in df.to_dict("records"):
    newline = deepcopy(line)
    del newline["model_name"]

    model_name = line.get("model_name")

    group_name, run_index = get_group_name_and_run_index_from_experiment_name(model_name)
    if not group_name:
        print(model_name)
        continue
    newline["model_name"] = group_name
    newline["run_index"] = run_index

    processed_dataset.append(newline)


processed_dataset = list(sorted(processed_dataset, key = lambda x: [x["dataset_name"], x["model_name"], x["run_index"]]))
newdf = pd.DataFrame(processed_dataset)

averagedf = compute_average_performance(newdf)
newdf = pd.concat([newdf, averagedf])

second_column = newdf.pop('model_name')
newdf.insert(1, 'model_name', second_column)

third_column = newdf.pop('run_index')
newdf.insert(2, 'run_index', third_column)

print(newdf)
newdf.to_csv("view_processed_12feb2025.csv", index=False)