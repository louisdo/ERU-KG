import random
import pandas as pd
from tqdm import tqdm

random.seed(42)

def read_tsv(path):
    dataset = []
    error_count = 0
    with open(path) as f:
        for line in tqdm(f):
            # if len(line) > 1:
            splitted = line.strip().split("\t")
                # if len(splitted) != 3: continue
            
            if len(splitted) != 3: 
                error_count += 1
                continue
            query, pos, neg = splitted  # first column is id

            dataset.append([query, pos, neg])

    print(f"Number of errors in dataset {path}:", error_count)
    return dataset



def combine_tsv(paths, output_size = 3000000, sample_size_each_dataset = 500000):
    dataset = []
    for path in paths:
        temp_dataset = read_tsv(path)
        dataset.extend(random.sample(temp_dataset, k = min(len(temp_dataset), sample_size_each_dataset)))

    print("Length of dataset", len(dataset))
    random.shuffle(dataset)
    return dataset
    # return random.sample(dataset, k = min(len(dataset), output_size))


def create_tsv(data, file_name):
    try:
        # Create a DataFrame from the data
        df = pd.DataFrame(data, columns=['query', 'pos', 'neg'])
        
        # Save the DataFrame to a TSV file
        df.to_csv(file_name, sep='\t', index=False, header = False, escapechar='\\')
        
        print(f"TSV file '{file_name}' has been created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    dataset = combine_tsv(paths = [
        "/home/lamdo/keyphrase_informativeness_test/splade/data/keyphrase_informative_scirepeval/triplets/raw.tsv", # query
        "/scratch/lamdo/unArxive/keyphrase_informativeness_unArxiv/triplets_full_paper_1citationpersentence_hardneg/raw.tsv", # cc
        # "/scratch/lamdo/unArxive/keyphrase_informativeness_unArxiv/triplets_title_abstract_hardneg/raw.tsv" # title
    ])


    create_tsv(dataset, file_name="/scratch/lamdo/unArxive/keyphrase_informativeness_combined_references/triplets_hardneg_no_titles/raw.tsv")