# Data processing

## Obtain the required data
We create ERU-KG's training dataset from
+ unarXive permissively licensed dataset: https://zenodo.org/records/7752615
+ arXiv dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv


We need to slightly process the downloaded arXiv dataset (the downloaded file is named `arxiv-metadata-oai-snapshot.json`). Run the following code to create `arxivid2metadata.json`
```python
import json
from tqdm import tqdm

ARXIV_DATASET_PATH = "/YOUR/DIRECTORY/arxiv-metadata-oai-snapshot.json"

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


with open("/YOUR/DIRECTORY/arxivid2metadata.json", "w") as f:
    json.dump(data, f)
```



## Run data processing code
To run data processing, please go to `data_processing.sh` and adjust the variables, then run the following command

```bash
chmod +x data_processing.sh
./data_processing.sh
```