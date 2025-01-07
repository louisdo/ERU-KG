import json, os, random, re
from tqdm import tqdm
from datasets import Dataset, DatasetDict

# set seed
random.seed(42)

ARXIV_DATASET_PATH = os.environ["ARXIV_DATASET_PATH"]
NUM_TRAIN = int(os.environ["NUM_TRAIN"])
NUM_TEST = int(os.environ["NUM_TEST"])
OUT_FOLDER = os.environ["OUT_FOLDER"]

CS_CATEGORIES = """cs.AI
cs.AR
cs.CC
cs.CE
cs.CG
cs.CL
cs.CR
cs.CV
cs.CY
cs.DB
cs.DC
cs.DL
cs.DM
cs.DS
cs.ET
cs.FL
cs.GL
cs.GR
cs.GT
cs.HC
cs.IR
cs.IT
cs.LG
cs.LO
cs.MA
cs.MM
cs.MS
cs.NA
cs.NE
cs.NI
cs.OH
cs.OS
cs.PF
cs.PL
cs.RO
cs.SC
cs.SD
cs.SE
cs.SI
cs.SY""".split("\n")
CS_CATEGORIES = [cat for cat in CS_CATEGORIES if cat]

def check_if_cs(doc):
    doc_cat = doc["categories"]
    for cat in CS_CATEGORIES:
        if cat in doc_cat:
            return True
    return False

def multi_label_one_hot_encode(categories, labels):
    one_hot = [0] * len(categories)
    
    for label in labels:
        if label in categories:
            index = categories.index(label)
            one_hot[index] = 1
    
    return one_hot

def slightly_process_text(text):
    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)
    return text


def process_dataset(train_dataset, test_dataset):
    dataset_test = DatasetDict({
        "train": Dataset.from_list([{"paper_id": line["id"], "label": multi_label_one_hot_encode(CS_CATEGORIES, [l for l in line["categories"].split(" ") if l])} for line in train_dataset]),
        "test": Dataset.from_list([{"paper_id": line["id"], "label": multi_label_one_hot_encode(CS_CATEGORIES, [l for l in line["categories"].split(" ") if l])} for line in test_dataset])
    })

    combined_dataset = train_dataset + test_dataset

    dataset = DatasetDict({
        "evaluation": Dataset.from_dict(
            {"doc_id": [line["id"] for line in combined_dataset], 
            "title": [line["title"] for line in combined_dataset],
            "abstract": [line["abstract"] for line in combined_dataset],
            "label": [line["categories"] for line in combined_dataset]}
            ),
    })

    return dataset_test, dataset



with open(ARXIV_DATASET_PATH) as f:
    dataset = []
    for i, line in enumerate(tqdm(f)):

        jline = json.loads(line)

        if check_if_cs(jline):
            dataset.append(jline)
        

TOTAL_NUM = NUM_TRAIN + NUM_TEST

sampled_dataset = random.sample(dataset, TOTAL_NUM)
train_dataset = sampled_dataset[:NUM_TRAIN]
test_dataset = sampled_dataset[NUM_TRAIN:]


dataset_test, dataset = process_dataset(train_dataset, test_dataset)


dataset_test.save_to_disk(os.path.join(OUT_FOLDER, "arxiv_classification_20k_test/"))
dataset.save_to_disk(os.path.join(OUT_FOLDER, "arxiv_classification_20k/"))

# with open(os.path.join(OUT_FOLDER, "0_train.json"), "w") as f:
#     json.dump(train_dataset, f)

# with open(os.path.join(OUT_FOLDER, "0_test.json"), "w") as f:
#     json.dump(test_dataset, f)