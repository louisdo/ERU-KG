import pandas as pd
from datasets import load_dataset


def process_triplets_dataset(triplet_dataset, 
                             split_name, 
                             sentence_field_name, 
                             positive_field_name, 
                             negative_field_name, 
                             lower = True):
    res = []
    for line in triplet_dataset[split_name]:
        sentence = line.get(sentence_field_name)
        positive = line.get(positive_field_name)
        negative = line.get(negative_field_name)

        processed_line = {
            "sentence": sentence,
            "positive": positive,
            "negative": negative
        }
        if lower:
            processed_line["sentence"] = processed_line["sentence"].lower()
            processed_line["positive"] = processed_line["positive"].lower()
            processed_line["negative"] = processed_line["negative"].lower()

        res.append(processed_line)

    return res

def create_tsv(data, filename):
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    
    # Ensure the correct column order
    column_order = ['sentence', 'positive', 'negative']
    df = df[column_order]
    
    # Write the DataFrame to a TSV file without header
    df.to_csv(filename, sep='\t', index=False, header=False)

triplet_dataset = []

pubmedqa_triplets = load_dataset("sentence-transformers/pubmedqa", "triplet-all")

# convert pubmedqa_triplets to standard format with 3 fields: sentence, positive, negative
temp = process_triplets_dataset(
    triplet_dataset=pubmedqa_triplets,
    split_name="train",
    sentence_field_name="anchor",
    positive_field_name="positive",
    negative_field_name="negative"
)
triplet_dataset.extend(temp)


specter_triplets = load_dataset("sentence-transformers/specter", "triplet")

temp = process_triplets_dataset(
    triplet_dataset=specter_triplets,
    split_name="train",
    sentence_field_name="anchor",
    positive_field_name="positive",
    negative_field_name="negative"
)
triplet_dataset.extend(temp)

create_tsv(triplet_dataset, filename="test.tsv")