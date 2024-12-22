import json, os
from datasets import load_dataset


KEYPHRASE_EXPANSION_PATH = os.getenv("KEYPHRASE_EXPANSION_PATH")
DATASET_NAME = os.environ["DATASET_NAME"]
OUTPUT_FILE = os.environ["OUTPUT_FILE"]
NUM_KEYPHRASES_EACH_TYPE = int(os.getenv("NUM_KEYPHRASES_EACH_TYPE", 10))


def process_dataset(dataset_name):
    if dataset_name == "scirepeval_fos_test":
        ds = load_dataset("allenai/scirepeval", "fos")
        ds_evaluation = ds["evaluation"]

    elif dataset_name == "scirepeval_mesh_descriptors_test":
        ds = load_dataset("allenai/scirepeval", "mesh_descriptors")
        ds_evaluation = ds["evaluation"]
    
    ds_evaluation = [dict(row) for row in ds_evaluation]

    print(type(ds_evaluation))
    
    return ds_evaluation


def add_keyphrase_expansion_to_dataset(ds_evaluation, 
                                       keyphrase_expansion_path = KEYPHRASE_EXPANSION_PATH,
                                       num_keyphrases_each_type = NUM_KEYPHRASES_EACH_TYPE):
    if not keyphrase_expansion_path: 
        print("No keyphrase expansion path provided")
        return ds_evaluation
    
    with open(keyphrase_expansion_path) as f:
        all_keyphrase_expansions = json.load(f)

    assert len(all_keyphrase_expansions) == len(ds_evaluation), [len(all_keyphrase_expansions), len(ds_evaluation)]

    for i in range(len(all_keyphrase_expansions)):
        keyphrase_expansion = all_keyphrase_expansions[i]
        present_keyphrases = keyphrase_expansion.get("automatically_extracted_keyphrases", {}).get("present_keyphrases", [])[:num_keyphrases_each_type]
        absent_keyphrases = keyphrase_expansion.get("automatically_extracted_keyphrases", {}).get("absent_keyphrases", [])[:num_keyphrases_each_type]

        kw_exp = ", ".join(present_keyphrases + absent_keyphrases)

        ds_evaluation[i]["keyphrase_expansion"] = kw_exp

    return ds_evaluation


if __name__ == "__main__":
    ds_evaluation = process_dataset(DATASET_NAME)
    ds_evaluation = add_keyphrase_expansion_to_dataset(ds_evaluation)
    

    with open(OUTPUT_FILE, "w") as f:
        json.dump(ds_evaluation, f)