import json
import pandas as pd
from utils import clean_text, simple_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords
from multiprocessing import Pool
from argparse import ArgumentParser

# INPUT_FILE = "/scratch/lamdo/unArxive/triplets_title_abstract_hardneg.json" # "/scratch/lamdo/unArxive/triplets_full_paper_1citationpersentence_hardneg.json"
# OUTPUT_FILE =  "/scratch/lamdo/unArxive/keyphrase_informativeness_unArxiv/triplets_title_abstract_hardneg/raw.tsv" # "/scratch/lamdo/unArxive/keyphrase_informativeness_unArxiv/triplets_full_paper_1citationpersentence_hardneg/raw.tsv"
STOPWORDS = set(list(stopwords.words('english')))

def create_tsv(data, file_name):
    try:
        # Create a DataFrame from the data
        df = pd.DataFrame(data, columns=['query', 'pos', 'neg'])
        
        # Save the DataFrame to a TSV file
        df.to_csv(file_name, sep='\t', index=False, header = False, escapechar='\\')
        
        print(f"TSV file '{file_name}' has been created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")



def process_triplet(line):
    query, pos, neg = line
    cleaned_triplet = [clean_text(query, deep_clean=True), 
                       clean_text(pos, deep_clean=True), 
                       clean_text(neg, deep_clean=True)]
    
    return cleaned_triplet

def main():

    parser = ArgumentParser()
    parser.add_argument("--input_file", type = str, required = True)
    parser.add_argument("--output_file", type = str, required = True)

    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    print("reading raw data")
    with open(input_file) as f:
        triplets_data_ = json.load(f)
    print("done reading. Processing")

    
    num_cores = 8

    # Create a pool of worker processes
    with Pool(num_cores) as pool:
        # Use tqdm to show a progress bar
        triplets_data = list(tqdm(
            pool.imap(process_triplet, triplets_data_),
            total=len(triplets_data_),
            desc="Processing"
        ))

    # Remove None values (triplets that didn't pass the filter)
    triplets_data = [triplet for triplet in triplets_data if triplet is not None]
    print(len(triplets_data))
    print(triplets_data[0])

    create_tsv(data = triplets_data, file_name=output_file)


if __name__ == "__main__":
    main()