import json
import pandas as pd
from utils import clean_text, simple_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords
from multiprocessing import Pool

INPUT_FILE = "/scratch/lamdo/unArxive/triplets_title_abstract_hardneg.json" # "/scratch/lamdo/unArxive/triplets_full_paper_1citationpersentence_hardneg.json"
OUTPUT_FILE =  "/scratch/lamdo/unArxive/keyphrase_informativeness_unArxiv/triplets_title_abstract_hardneg/raw.tsv" # "/scratch/lamdo/unArxive/keyphrase_informativeness_unArxiv/triplets_full_paper_1citationpersentence_hardneg/raw.tsv"
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


def filter_data_sample(data_sample, min_anchor_length = 4, min_overlapping_with_pos = 3):
    # data_sample should be a list of 3 items, query, pos, neg
    # make sure that the query has > min_anchor_length words
    # make sure that the query has > min_overlapping_with_pos words overlapping with positive sample (skip the stopwords)


    query, pos, neg = data_sample

    # query_tokens = simple_tokenize(query, lower = True)
    # pos_tokens = simple_tokenize(pos, lower = True)
    # neg_tokens = simple_tokenize(neg, lower = True)

    # if len(query_tokens) < min_anchor_length: return False

    # query_tokens_without_stopwords = [tok for tok in query_tokens if tok not in STOPWORDS]
    # pos_tokens_without_stopwords = [tok for tok in pos_tokens if tok not in STOPWORDS]

    # if not query_tokens_without_stopwords or not pos_tokens_without_stopwords:
    #     return False

    # if len(set(query_tokens_without_stopwords).intersection(set(pos_tokens_without_stopwords))) < min_overlapping_with_pos:
    #     return False
    
    return True



def process_triplet(line):
    query, pos, neg = line
    cleaned_triplet = [clean_text(query, deep_clean=True), 
                       clean_text(pos, deep_clean=True), 
                       clean_text(neg, deep_clean=True)]
    
    is_good_triplet = filter_data_sample(data_sample=cleaned_triplet)
    
    if is_good_triplet:
        return cleaned_triplet
    return None

def main():
    print("reading raw data")
    with open(INPUT_FILE) as f:
        triplets_data_ = json.load(f)
    print("done reading. Processing")
    # Determine the number of CPU cores to use
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

    create_tsv(data = triplets_data, file_name=OUTPUT_FILE)


if __name__ == "__main__":
    main()