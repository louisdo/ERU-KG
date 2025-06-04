import json, os, random, re
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from tqdm import tqdm
from utils import clean_text


FOLDER_NAME = "/scratch/lamdo/unArxive"
ARXIVID2METADATA_FILE = "/scratch/lamdo/arxiv_dataset/arxivid2metadata.json"
OUTPUT_FILE=  "/scratch/lamdo/unArxive/triplets_title_abstract_hardneg.json"

NUMBER_NEGATIVES_WITHIN_PAPER = 3
# NUMBER_NEGATIVES_RANDOM_SAMPLING = 3




with open(ARXIVID2METADATA_FILE) as f:
    _data = json.load(f)
    print(len(_data))

    ARXIV_ID_TO_METADATA = {}
    for arxiv_id, metadata in  tqdm(_data.items()):
        title = clean_text(metadata.get("title"))
        abstract = clean_text(metadata.get("abstract"))

        ARXIV_ID_TO_METADATA[arxiv_id] = {"title": title, "abstract": abstract}

    ARXIV_IDS = list(ARXIV_ID_TO_METADATA.keys())


def get_citation_id_by_section(datapoint):
    res = {}

    body_text = datapoint.get("body_text", [])

    for section in body_text:
        section_name = section.get("section")
        if section_name not in res: res[section_name] = set([])

        cite_spans = section.get("cite_spans", [])

        res[section_name].update([cspan["ref_id"] for cspan in cite_spans])

    return {k:list(v) for k, v in res.items()}



def check_if_arxive_id_exists_in_cite_span(ref_id, bib_entries):
    return ref_id and \
            bib_entries.get(ref_id) and \
            bib_entries.get(ref_id).get("ids") and \
                bib_entries.get(ref_id).get("ids").get("arxiv_id")


def get_arxive_id_from_bib_entries(ref_id, bib_entries):
    try:
        return bib_entries.get(ref_id).get("ids").get("arxiv_id")
    except Exception as e:
        return None


def create_triplets_from_datapoint(datapoint, min_section_length = 10):

    res = []
    citation_id_by_section = get_citation_id_by_section(datapoint)

    bib_entries = datapoint.get("bib_entries", {})


    # included_sections = ['introduction', "related work", "related works"]

    for section in citation_id_by_section:
        # if not any([section_name in section.lower() for section_name in included_sections]): continue

        citations_ref_ids = citation_id_by_section[section]
        if not citations_ref_ids: continue

        citations_mentioned_in_other_sections = set([citation_ref_id for other_section_name, citations_ref_ids_other_section in citation_id_by_section.items() if other_section_name != section for citation_ref_id in citations_ref_ids_other_section])
        citations_mentioned_in_other_sections = list([item for item in citations_mentioned_in_other_sections if item not in citations_ref_ids])

        # build triplets
        for citation_ref_id in citations_ref_ids:
            if not check_if_arxive_id_exists_in_cite_span(citation_ref_id, bib_entries): continue

            citation_arxive_id = get_arxive_id_from_bib_entries(citation_ref_id, bib_entries)

            if citation_arxive_id not in ARXIV_ID_TO_METADATA: continue

            # these two will form positive pairs
            citation_title = ARXIV_ID_TO_METADATA[citation_arxive_id].get("title")
            citation_abstract = ARXIV_ID_TO_METADATA[citation_arxive_id].get("abstract")


            # now get the negatives, first, get negatives within the same paper
            citations_mentioned_in_other_sections_sampled = random.sample(citations_mentioned_in_other_sections, min(NUMBER_NEGATIVES_WITHIN_PAPER, len(citations_mentioned_in_other_sections)))
            arxive_ids_of_citations_mentioned_in_other_sections_sampled = [get_arxive_id_from_bib_entries(citation_ref_id, bib_entries) for citation_ref_id in citations_mentioned_in_other_sections_sampled]
            arxive_ids_of_citations_mentioned_in_other_sections_sampled = [item for item in arxive_ids_of_citations_mentioned_in_other_sections_sampled if item]
            arxive_ids_of_citations_mentioned_in_other_sections_sampled = [item for item in arxive_ids_of_citations_mentioned_in_other_sections_sampled if item in ARXIV_ID_TO_METADATA]
            neg_samples_within_paper = [ARXIV_ID_TO_METADATA[other_section_arxive_id]["abstract"] for other_section_arxive_id in arxive_ids_of_citations_mentioned_in_other_sections_sampled]
            
            # then get random negatives
            # random_sampled_arxive_ids = random.sample(ARXIV_IDS, min(NUMBER_NEGATIVES_RANDOM_SAMPLING, len(ARXIV_IDS)))
            # neg_samples_random = [ARXIV_ID_TO_METADATA[arxiv_id]["abstract"] for arxiv_id in random_sampled_arxive_ids]

            # combine neg samples
            neg_samples = neg_samples_within_paper #+ neg_samples_random
            for neg in neg_samples:
                res.append([citation_title, citation_abstract, neg])


    return res


if __name__ == "__main__":
    folders = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", 
           "10", "11", "12", "13", "14", "15", "16", "17", "18", 
           "19", "20", "21", "22", "93", "97", "98"]
    folders = [os.path.join(FOLDER_NAME, folder) for folder in folders]

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
            line
        )

        res.extend(line_triplets)

    print(len(res))
    with open(OUTPUT_FILE, "w") as f:
        json.dump(res, f, indent=4)