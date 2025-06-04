import random
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

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


def main():
    parser = ArgumentParser()
    parser.add_argument("--queries_data_file", type = str, required = True)
    parser.add_argument("--citation_contexts_data_file", type = str, required=True)
    parser.add_argument("--titles_data_file", type = str, required = True)
    parser.add_argument("--output_file", type = str, required = True)

    args = parser.parse_args()

    queries_data_file = args.queries_data_file
    citation_contexts_data_file = args.citation_contexts_data_file
    titles_data_file = args.titles_data_file
    output_file = args.output_file

    dataset = combine_tsv(paths = [
        queries_data_file,
        citation_contexts_data_file,
        titles_data_file
    ])


    create_tsv(dataset, file_name=output_file)

if __name__ == "__main__":
    main()