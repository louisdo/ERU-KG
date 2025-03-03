import json, os, nltk, sys
sys.path.append("../")
from process_dataset import process_dataset
from collections import Counter
from tqdm import tqdm
from functools import lru_cache

KG_RESULTS_FOLDER = "/scratch/lamdo/precompute_keyphrase_extraction"

# datasets = ["scifact_queries", "scidocs_queries", "trec_covid_queries", "nfcorpus_queries", "doris_mae_queries", "acm_cr_queries"]
datasets = ["scifact", "scidocs", "trec_covid", "nf_corpus", "doris_mae", "acm_cr"][:2]
models = ["autokeygen-1", "copyrnn-1", "uokg-1", "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-1_position_penalty+length_penalty", "tpg-1"][:-1]




PORTER_STEMMER = nltk.stem.PorterStemmer()
@lru_cache(maxsize=100000)
def stem_func(token):
    return PORTER_STEMMER.stem(token)

def check_absent_keyphrase_type(text, absent_keyphrases):
    # 0: reordered, 1: mix, 2: unseen

    text_tokens = nltk.tokenize.word_tokenize(text.lower())
    stemmed_text_tokens = set([stem_func(token) for token in text_tokens])
    stemmed_text_tokens_joined = " ".join(stemmed_text_tokens)

    absent_keyphrases_types = []
    absent_terms_added = set([])
    terms_added = set([])
    for absent_keyphrase in absent_keyphrases:
        absent_keyphrase_tokens = nltk.tokenize.word_tokenize(absent_keyphrase)
        stemmed_absent_keyphrase_tokens = set([stem_func(token) for token in absent_keyphrase_tokens])

        overlap_keyphrase_text = set([tok for tok in stemmed_absent_keyphrase_tokens if tok in stemmed_text_tokens_joined])
        absent_terms = stemmed_absent_keyphrase_tokens.difference(overlap_keyphrase_text)
        absent_terms_added.update(absent_terms)
        terms_added.update(stemmed_absent_keyphrase_tokens)

        if len(overlap_keyphrase_text) == len(absent_keyphrase_tokens): 
            absent_keyphrases_types.append(0)
        elif len(absent_keyphrase_tokens) > len(overlap_keyphrase_text) > 0: 
            absent_keyphrases_types.append(1)
        else: 
            absent_keyphrases_types.append(2)

    return absent_keyphrases_types, len(absent_terms_added), len(terms_added)



record = {}
number_of_absent_terms_added = {}
total = {}
number_of_inputs = 0
for dataset in datasets[:]:
    record[dataset] = {}
    ds = process_dataset(dataset)
    number_of_inputs += len(ds)
    for model in models:
        record[dataset][model] = Counter()

        if not number_of_absent_terms_added.get(model):
            number_of_absent_terms_added[model] = 0
            total[model] = 0

        file_name = f"{dataset}--{model}.json"
        file_path = os.path.join(KG_RESULTS_FOLDER, file_name)

        with open(file_path, "r") as f:
            data = json.load(f)

        top_k = 5 if "queries" in dataset else 10
        all_absent_keyphrases = [item["automatically_extracted_keyphrases"]["present_keyphrases"][:top_k] + item["automatically_extracted_keyphrases"]["absent_keyphrases"][:top_k] for item in data]
        
        assert len(ds) == len(all_absent_keyphrases)
        for i in tqdm(range(len(ds)), desc=f"{dataset}--{model}"):
            line = ds[i]
            absent_keyphrases = all_absent_keyphrases[i]

            temp = list(check_absent_keyphrase_type(line["text"], absent_keyphrases))
            absent_keyphrase_type = temp[0]
            num_absent_terms_added = temp[1]
            num_terms = temp[2]

            number_of_absent_terms_added[model] += num_absent_terms_added
            total[model] += num_terms

            record[dataset][model].update(Counter(absent_keyphrase_type))


# with open("absent_keyphrase_distribution_result.json", "w") as f:
#     json.dump(record, f, indent = 4)


with open("avg_absent_terms_added.json", "w") as f:
    json.dump({k:round(v / number_of_inputs,1) for k,v in number_of_absent_terms_added.items()}, f, indent = 4)