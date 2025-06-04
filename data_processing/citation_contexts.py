import json, os, random, re
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from tqdm import tqdm
from utils import clean_text

FOLDER_NAME = "/scratch/lamdo/unArxive"
ARXIVID2METADATA_FILE = "/scratch/lamdo/arxiv_dataset/arxivid2metadata.json"
OUTPUT_FILE = "/scratch/lamdo/unArxive/triplets_full_paper_1citationpersentence_hardneg.json"

NUMBER_NEGATIVES_WITHIN_PAPER = 3
# NUMBER_NEGATIVES_RANDOM_SAMPLING = 3

MAX_CITATIONS_PER_SENTENCE = 1
MIN_SECTION_LENGTH = 10



with open(ARXIVID2METADATA_FILE) as f:
    _data = json.load(f)
    print(len(_data))

    ARXIV_ID_TO_TEXT = {}
    for arxiv_id, metadata in  tqdm(_data.items()):
        title = clean_text(metadata.get("title"))
        abstract = clean_text(metadata.get("abstract"))

        text = f"{title}. {abstract}"
        ARXIV_ID_TO_TEXT[arxiv_id] = text

    ARXIV_IDS = list(ARXIV_ID_TO_TEXT.keys())

def count_citation_groups(text):
    citation_group_pattern = r'({{cite:[0-9a-f]{40}}}(?:\s*,\s*{{cite:[0-9a-f]{40}}})*)'
    matches = re.findall(citation_group_pattern, text)
    return len(matches)


def create_triplets_from_datapoint(datapoint, 
                                   min_section_length = 10,
                                   max_citation_in_sentence = None):
    body_text = datapoint.get("body_text", [])
    bib_entries = datapoint.get("bib_entries", {})
    if not body_text or not bib_entries: return []

    res = []

    for section in body_text:
        # if not isinstance(section.get("section"), str) or not any([item == section.get("section").lower() for item in include_section]):
        #     continue


        # section_name = section.get("section")
        # if not isinstance(section_name, str): continue
        # section_name = section_name.lower()

        # if section_name not in ['introduction', 'related work', "related works"]: continue
        # include_section = ['introduction', 'related work', "related works"]
        # if not any([item in section_name for item in include_section]): continue

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
            neg_citation_within_paper = random.sample(unmentioned_cite_spans, k = min(len(unmentioned_cite_spans), NUMBER_NEGATIVES_WITHIN_PAPER))
            # neg_citation_within_paper = []
            # neg_citation_random_sampling = random.sample(ARXIV_IDS, k = NUMBER_NEGATIVES_RANDOM_SAMPLING)
            # neg_citation_random_sampling = []
            neg_citation = neg_citation_within_paper #+ neg_citation_random_sampling
            neg_citation = [ARXIV_ID_TO_TEXT[item] for item in neg_citation if item in ARXIV_ID_TO_TEXT]

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

            pos_citation = [ARXIV_ID_TO_TEXT[item] for item in list((set(pos_citation))) if item in ARXIV_ID_TO_TEXT]


            for pos in pos_citation:
                for neg in neg_citation:
                    res.append([sentence, pos, neg])
    return res


if __name__ == "__main__":
    folders = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", 
           "10", "11", "12", "13", "14", "15", "16", "17", "18", 
           "19", "20", "21", "22", "93", "97", "98"]
    folders = [os.path.join(FOLDER_NAME, folder) for folder in folders]

    # res = []
    # all_file_data = []
    # for folder in tqdm(folders[:10], desc = "Reading files"):
    #     files = os.listdir(folder)
    #     files_full_paths = [os.path.join(folder, file) for file in files]

    #     for file_full_path in tqdm(files_full_paths):
    #         with open(file_full_path) as f:
    #             for line in f:
    #                 all_file_data.append(json.loads(line))


    # for line in tqdm(all_file_data, desc = "Processing"):
    #     line_triplets = create_triplets_from_datapoint(
    #         line, 
    #         min_section_length=MIN_SECTION_LENGTH, 
    #         max_citation_in_sentence=MAX_CITATIONS_PER_SENTENCE
    #     )

    #     res.extend(line_triplets)

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
            min_section_length=MIN_SECTION_LENGTH, 
            max_citation_in_sentence=MAX_CITATIONS_PER_SENTENCE
        )

        res.extend(line_triplets)

    print(len(res))
    with open(OUTPUT_FILE, "w") as f:
        json.dump(res, f, indent=4)