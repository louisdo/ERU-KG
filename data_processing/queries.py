import random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from argparse import ArgumentParser
# output_file = "/home/lamdo/keyphrase_informativeness_test/splade/data/keyphrase_informative_scirepeval/triplets/raw.tsv"

def create_tsv(data, file_name):
    try:
        # Create a DataFrame from the data
        df = pd.DataFrame(data, columns=['query', 'pos', 'neg'])
        
        # Save the DataFrame to a TSV file
        df.to_csv(file_name, sep='\t', index=False, header = False, escapechar='\\')
        
        print(f"TSV file '{file_name}' has been created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

def slightly_process_text(text):
    return text.replace("\n", "").replace("\t", "")

def main():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type = str, required = True)
    parser.add_argument("--max_data", type = int, default = 3000000)

    args = parser.parse_args()

    output_file = args.output_file
    max_data = args.max_data

    ds = load_dataset("allenai/scirepeval", "search")


    processed_data = []
    for line in tqdm(ds["train"], desc = "Get data from train split"):
        query = line.get("query")

        if not isinstance(query, str): 
            if isinstance(query, dict):
                query_title = slightly_process_text(query.get("title"))
                query_abstract = slightly_process_text(query.get("abstract"))

                query = f"{query_title.lower()}. {query_abstract.lower()}"
            else: continue

        query = slightly_process_text(query.lower())
        candidates = line.get("candidates")

        positives = []
        negatives = []

        for cand in candidates:
            score = cand.get("score")
            title = slightly_process_text(cand.get("title"))
            abstract = slightly_process_text(cand.get("abstract"))

            text = f"{title.lower()}. {abstract.lower()}"

            if score == 0:
                negatives.append(text)
            elif score > 0: 
                positives.append(text)

        if not positives or not negatives: continue

        for pos in positives:
            if not pos: continue
            for neg in negatives:
                if not neg: continue
                processed_data.append([query, pos, neg])

    print(f"There are {len(processed_data)} rows")
    create_tsv(data = processed_data[:max_data], file_name = output_file)