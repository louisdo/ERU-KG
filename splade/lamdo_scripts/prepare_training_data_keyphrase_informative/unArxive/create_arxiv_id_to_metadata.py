# import os, json
# from tqdm import tqdm

# FOLDER_NAME = "/scratch/lamdo/unArxive"

# if __name__ == "__main__":

#     folders = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", 
#            "10", "11", "12", "13", "14", "15", "16", "17", "18", 
#            "19", "20", "21", "22", "93", "97", "98"]
#     folders = [os.path.join(FOLDER_NAME, folder) for folder in folders]

#     res = {}
#     for folder in tqdm(folders[:]):
#         files = os.listdir(folder)
#         files_full_paths = [os.path.join(folder, file) for file in files]

#         for file_full_path in files_full_paths:

#             file_data = []
#             with open(file_full_path) as f:
#                 for line in f:
#                     file_data.append(json.loads(line))

#             paper_ids = [line.get("paper_id") for line in file_data]
#             paper_metadatas = [line.get("metadata") for line in file_data]

#             for paper_id, paper_metadata in zip(paper_ids, paper_metadatas):
#                 res[paper_id] = paper_metadata
#         print("Current number of papers:", len(res))

#     with open("/scratch/lamdo/unArxive/arxivid2metadata.json", "w") as f:
#         json.dump(res, f, indent=4)


import json
from tqdm import tqdm

ARXIV_DATASET_PATH = "/scratch/lamdo/arxiv_dataset/arxiv-metadata-oai-snapshot.json"

with open(ARXIV_DATASET_PATH) as f:
    data = {}
    for line in tqdm(f): 
        jline = json.loads(line)

        arxiv_id = jline.get("id")
        title = jline.get("title")
        abstract = jline.get("abstract")

        metadata = {
            "title": title,
            "abstract": abstract
        }

        data[arxiv_id] = metadata


with open("/scratch/lamdo/arxiv_dataset/arxivid2metadata.json", "w") as f:
    json.dump(data, f)