import json, os, random, re
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from tqdm import tqdm
from utils import clean_text
from argparse import ArgumentParser

# FOLDER_NAME = "/scratch/lamdo/unArxive"
# ARXIVID2METADATA_FILE = "/scratch/lamdo/arxiv_dataset/arxivid2metadata.json"
# OUTPUT_FILE = "/scratch/lamdo/unArxive/triplets_full_paper_1citationpersentence_hardneg.json"

# NUMBER_NEGATIVES_WITHIN_PAPER = 3
# NUMBER_NEGATIVES_RANDOM_SAMPLING = 3

# MAX_CITATIONS_PER_SENTENCE = 1
# MIN_SECTION_LENGTH = 10

CONFIGS = {
    "number_negatives_within_paper": 3,
    "max_citations_per_sentence": 1,
    "min_section_length": 10
}

def count_citation_groups(text):
    citation_group_pattern = r'({{cite:[0-9a-f]{40}}}(?:\s*,\s*{{cite:[0-9a-f]{40}}})*)'
    matches = re.findall(citation_group_pattern, text)
    return len(matches)


def create_triplets_from_datapoint(datapoint, 
                                   arxiv_id_to_text: dict,
                                   min_section_length = 10,
                                   max_citation_in_sentence = None):
    body_text = datapoint.get("body_text", [])
    bib_entries = datapoint.get("bib_entries", {})
    if not body_text or not bib_entries: return []

    res = []

    for section in body_text:

        section_text = section.get("text")
        if not isinstance(section_text, str) or len(section_text.split(" ")) < min_section_length:
            continue


        cite_spans = section.get("cite_spans", [])
        ref_spans = section.get("ref_spans", [])

        unmentioned_cite_spans = []
        mentioned_cite_ref_ids = set([item.get("ref_id") for item in cite_spans])
        for bib_entry_id, bib_entry_value in bib_entries.items():
            if bib_entry_id in mentioned_cite_ref_ids or not (bib_entry_value and bib_entry_value.get("ids") and bib_entry_value.get("ids").get("arxiv_id")): continue
            unmentioned_cite_spans.append(bib_entry_value.get("ids").get("arxiv_id"))



        for rs in ref_spans:
            rs_text = rs.get("text")
            section_text = section_text.replace(rs_text, "")

        sentences = nltk_sent_tokenize(section_text)

        for _sentence in sentences:
            sentence = _sentence

            # citations_in_sentences = len(set([item.get("text") for item in cite_spans if item.get("text") in sentence]))
            citation_groups_in_sentence = count_citation_groups(_sentence)
            if max_citation_in_sentence and citation_groups_in_sentence > max_citation_in_sentence: continue

            pos_citation = []
            neg_citation_within_paper = random.sample(unmentioned_cite_spans, k = min(len(unmentioned_cite_spans), CONFIGS["number_negatives_within_paper"]))
            # neg_citation_within_paper = []
            # neg_citation_random_sampling = random.sample(ARXIV_IDS, k = NUMBER_NEGATIVES_RANDOM_SAMPLING)
            # neg_citation_random_sampling = []
            neg_citation = neg_citation_within_paper #+ neg_citation_random_sampling
            neg_citation = [arxiv_id_to_text[item] for item in neg_citation if item in arxiv_id_to_text]

            for cite_span in cite_spans:
                cite_span_text = cite_span.get("text")
                sentence = sentence.replace(cite_span_text, "")

                if not (cite_span.get("ref_id") and \
                       bib_entries.get(cite_span.get("ref_id")) and \
                        bib_entries.get(cite_span.get("ref_id")).get("ids") and \
                            bib_entries.get(cite_span.get("ref_id")).get("ids").get("arxiv_id")): continue
                
                cite_arxiv_id = bib_entries.get(cite_span.get("ref_id")).get("ids").get("arxiv_id")
                if cite_span_text not in _sentence:
                    continue

                pos_citation.append(cite_arxiv_id)

            pos_citation = [arxiv_id_to_text[item] for item in list((set(pos_citation))) if item in arxiv_id_to_text]


            for pos in pos_citation:
                for neg in neg_citation:
                    res.append([sentence, pos, neg])
    return res

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, required = True, help = "UnArxiv folder")
    parser.add_argument("--arxive_metadata_file", type = str, required = True, help = "Path to arxive metadata")
    parser.add_argument("--output_file", type = str, required = True, help = "Output file")
    parser.add_argument("--number_negatives_within_paper", type = int, default = 3)
    parser.add_argument("--max_citations_per_sentence", type = int, default = 1)
    parser.add_argument("--min_section_length", type = int, default = 10)

    args = parser.parse_args()

    input_folder = args.input_folder
    arxive_metadata_file = args.arxive_metadata_file
    output_file = args.output_file
    number_negatives_within_paper = args.number_negatives_within_paper
    max_citations_per_sentence = args.max_citations_per_sentence
    min_section_length = args.min_section_length

    CONFIGS["number_negatives_within_paper"] = number_negatives_within_paper
    CONFIGS["max_citations_per_sentence"] = max_citations_per_sentence
    CONFIGS["min_section_length"] = min_section_length

    folders = os.listdir(input_folder)
    folders = [os.path.join(input_folder, folder) for folder in folders]

    with open(arxive_metadata_file) as f:
        _data = json.load(f)

        ARXIV_ID_TO_TEXT = {}
        for arxiv_id, metadata in  tqdm(_data.items()):
            title = clean_text(metadata.get("title"))
            abstract = clean_text(metadata.get("abstract"))

            text = f"{title}. {abstract}"
            ARXIV_ID_TO_TEXT[arxiv_id] = text

    res = []
    all_file_data = []
    for folder in tqdm(folders[:], desc = "Reading files"):
        files = os.listdir(folder)
        files_full_paths = [os.path.join(folder, file) for file in files]

        for file_full_path in tqdm(files_full_paths):
            with open(file_full_path) as f:
                for line in f:
                    all_file_data.append(json.loads(line))


    for line in tqdm(all_file_data, desc = "Processing"):
        line_triplets = create_triplets_from_datapoint(
            line, 
            arxiv_id_to_text=ARXIV_ID_TO_TEXT,
            min_section_length=CONFIGS["min_section_length"], 
            max_citation_in_sentence=CONFIGS["max_citations_per_sentence"]
        )

        res.extend(line_triplets)

    with open(output_file, "w") as f:
        json.dump(res, f, indent=4)




if __name__ == "__main__":
    main()