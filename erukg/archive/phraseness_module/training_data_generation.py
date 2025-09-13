import json, hashlib, sys, random
sys.path.append("..")
from nounphrase_extractor import CandidateExtractorRegExpNLTK
from tqdm import tqdm
from functools import lru_cache


# random.seed(42)

CAP = 500000


# we will start from a triplet file (any will do), the triplet file should be a tsv file
TRIPLET_FILE = "/scratch/lamdo/unArxive/keyphrase_informativeness_combined_references/triplets_v2/raw.tsv"
OUTPUT_FILE = "/scratch/lamdo/phraseness_module/data/data200k.json"


CANDEXT = CandidateExtractorRegExpNLTK([1,5])

# @lru_cache(10000)
def hash_text(text):
    """Generates a SHA-256 hash for the given text."""
    hasher = hashlib.sha256()
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()




if __name__ == "__main__":
    with open(TRIPLET_FILE) as f:
        visited = set([])
        training_data_for_phraseness_module = []
        count = 0
        for line in tqdm(f):
            count += 1
            if count > CAP: break
            splitted_line = line.split("\t")

            if len(splitted_line) != 3:
                print("Oh no")
                continue

            query, pos, _ = splitted_line
            hash_id = hash_text(f"{query}|{pos}")

            if hash_id in visited: continue

            nounphrases_in_reference = CANDEXT(query)
            nounphrases_in_text = CANDEXT(pos)

            if not nounphrases_in_reference or not nounphrases_in_text: continue

            for nphrase in list(set(nounphrases_in_reference + nounphrases_in_text)):
                training_data_for_phraseness_module.append([pos, nphrase])

            visited.add(hash_id)


    print("Training data size", len(training_data_for_phraseness_module))
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(training_data_for_phraseness_module, f)