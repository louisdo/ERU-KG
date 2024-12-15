import json, os, shutil, nltk, re
from collections import Counter
from tqdm import tqdm


class CandidateExtractorRegExpNLTK:
    """
    CandidateExtractorRegExpNLTK class extracts candidate phrases from a given text using regular expressions
    and Natural Language Toolkit (NLTK) tools.

    Args:
    - length_range (list, optional): A list specifying the minimum and maximum length of candidate phrases.
      Default is [2, 5].
    - cand_counter (Counter, optional): A Counter object to track the occurrences of candidate phrases.
      Default is None.

    Attributes:
    - np_parser (nltk.RegexpParser): A RegexpParser object for parsing noun phrases based on the specified grammar.
    - length_range (list): A list specifying the minimum and maximum length of candidate phrases.
    - cand_counter (Counter): A Counter object to track the occurrences of candidate phrases.

    Methods:
    - word_check(word): Checks if a word is valid based on a set of conditions.
    - get_best_subphrase(tokens): Retrieves the best subphrase from a list of tokens based on the candidate counter.
    - get_phrase_from_subtree(subtree): Extracts a candidate phrase from a given subtree.
    - __call__(text): Extracts candidate phrases from the input text.

    Examples:
    >>> extractor = CandidateExtractorRegExpNLTK(length_range=[2, 5])
    >>> text = "This is a sample text with candidate phrases."
    >>> candidates = extractor(text)
    """

    def __init__(self, length_range=[2, 5], cand_counter=None):
        GRAMMAR_EN = r"""
        TT: {(<NN.*|JJ.*>+<NN.*|CD>)|<NN.*>}
        """
        self.np_parser = nltk.RegexpParser(GRAMMAR_EN)
        self.length_range = length_range
        self.cand_counter = cand_counter

    @staticmethod
    def word_check(word):
        if len(word) > 1 and re.search(r"^([a-z]+-)*[a-z]+(\s[0-9]+){0,1}$", word):
            return True
        return False

    def get_best_subphrase(self, tokens):
        all_subphrases = []
        for length in range(2, self.length_range[1] + 1):
            all_subphrases.extend(
                [
                    " ".join(tokens[index : index + length])
                    for index in range(len(tokens) - length + 1)
                ]
            )
        all_subphrases = [p for p in all_subphrases if self.cand_counter[p] > 0]

        if len(all_subphrases) == 0:
            return []
        res = max(all_subphrases, key=lambda x: self.cand_counter[x])

        return res.split(" ")

    def get_phrase_from_subtree(self, subtree):
        exp = []
        flag = True
        for l in subtree.leaves():
            w = str(l[0])
            if not self.word_check(w):
                flag = False
                break
            exp.append(w)
        if len(exp) > self.length_range[1]:
            if not self.cand_counter:
                flag = False
            else:
                exp = self.get_best_subphrase(exp)
                if len(exp) == 0:
                    flag = False
            exp = exp[-self.length_range[1] :]
        phrase_length = len(exp)
        exp = " ".join(exp)
        if flag:
            return exp, phrase_length
        return None, 0

    def __call__(self, text):
        tokenized = nltk.word_tokenize(text)
        text_token_counter = Counter(tokenized)
        tags = nltk.pos_tag(tokenized)

        tree = self.np_parser.parse(tags)

        cands = []
        cands_counter = Counter()
        for st in list(tree.subtrees(filter=lambda x: x.label().endswith("TT"))):
            if len(st.leaves()) < self.length_range[0]:
                continue

            cand, cand_length = self.get_phrase_from_subtree(st)
            if cand is None:
                continue
            cands_counter[cand] += 1
            if cands_counter[cand] == 1:
                cands.append(cand)
            # if cand_length >= 2:
            #     if cands_counter[cand] == 1: cands.append(cand) # change to == if you dont want multiset
            # else:
            #     if cands_counter[cand] == 2: cands.append(cand) # change to == if you dont want multiset

        return list(cands)


CANDEXT = {"candext": None}

def remove_folder(folder_path):
    """
    Removes a folder and all its contents.
    
    Args:
    folder_path (str): The path to the folder to be removed.
    
    Returns:
    bool: True if the folder was successfully removed, False otherwise.
    """
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Remove the folder and all its contents
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' has been successfully removed.")
            return True
        else:
            print(f"The folder '{folder_path}' does not exist.")
            return False
    except Exception as e:
        print(f"An error occurred while trying to remove the folder: {e}")
        return False

def maybe_create_folder(folder_path):
    """
    Check if a folder exists and create it if it doesn't.

    Args:
    folder_path (str): The path of the folder to be created.

    Returns:
    bool: True if the folder was created, False if it already existed.
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
            return True
        except OSError as e:
            print(f"Error creating folder {folder_path}: {e}")
            return False
    else:
        print(f"Folder already exists: {folder_path}")
        return False
    
def delete_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' has been deleted successfully.")
            return True
        else:
            print(f"Folder '{folder_path}' does not exist.")
            return False
    except PermissionError:
        print(f"Permission denied to delete the folder '{folder_path}'.")
        return False
    except Exception as e:
        print(f"An error occurred while deleting the folder: {e}")
        return False


def duplicate_top_keywords(keywords):
    chunk_length = int(len(keywords) / 3)

    res = []
    for i in range(0, len(keywords)):
        if 0 <= i < chunk_length:
            to_extend = [keywords[i]] * 3
        elif chunk_length <= i < chunk_length * 2:
            to_extend = [keywords[i]] * 2
        else:
            to_extend = [keywords[i]]

        res.extend(to_extend)

    return res


def convert_dataset_to_pyserini_format(dataset, 
                                       ids, 
                                       keyword_for_document_expansion_path, 
                                       documents_vectors = None,
                                       convert_nounphrases_to_question = False,
                                       extract_nounphrases = False,
                                       apply_expansion_using_keyword = True,
                                       num_keyword_each_type = 10,
                                       expansion_only_present_keyword = False):
    
    if extract_nounphrases:
        if CANDEXT["candext"] == None:
            CANDEXT["candext"] = CandidateExtractorRegExpNLTK([2, 5])

    document_vector_status = isinstance(documents_vectors, list) and len(documents_vectors) == len(dataset)

    # dataset is expected to be a list of pairs of strings
    assert isinstance(dataset, list)

    if keyword_for_document_expansion_path:
        with open(keyword_for_document_expansion_path) as f:
            expansion_keywords = json.load(f)

        assert len(expansion_keywords) == len(dataset)
    else:
        expansion_keywords = [{} for _ in range(len(dataset))]

    documents = []
    for i, line in enumerate(tqdm(dataset)):
        text = line

        if extract_nounphrases:
            nounphrases = CANDEXT["candext"](text)

        automatically_extracted_keyphrases = expansion_keywords[i].get("automatically_extracted_keyphrases")
        if isinstance(automatically_extracted_keyphrases, dict):
            present_keyphrases = automatically_extracted_keyphrases.get("present_keyphrases")
            absent_keyphrases = automatically_extracted_keyphrases.get("absent_keyphrases")

            _present_keyphrases = [item for item in present_keyphrases if "<tgt_unk>" not in item]
            _absent_keyphrases = [item for item in absent_keyphrases if "<tgt_unk>" not in item]
            if expansion_only_present_keyword:
                kw_exp = _present_keyphrases[:num_keyword_each_type]
            else:
                kw_exp = _present_keyphrases[:num_keyword_each_type] + _absent_keyphrases[:num_keyword_each_type]
        else:
            kw_exp = None
            present_keyphrases = []

        kw_exp = kw_exp if kw_exp else []
        # kw_exp = duplicate_top_keywords(kw_exp[:5]) + kw_exp[5:]
        if convert_nounphrases_to_question:
            kw_exp = [f"what is {item}?" for item in kw_exp]

        if kw_exp and apply_expansion_using_keyword:
            kw_exp_string = ", ".join(kw_exp[:])
            kw_exp_string = f"\n\n{kw_exp_string}"
        else: kw_exp_string = ""

        all_contents = [text, kw_exp_string]
        content = "\n".join([item for item in all_contents if item])
        # content = text if not kw_exp_string else f"{text}\n{kw_exp_string}"

        if extract_nounphrases:
            to_append = {"id": ids[i], 
                              "contents": content, 
                              "nounphrases": nounphrases, 
                              "present_keyphrases": []}
        else:
            to_append = {"id": ids[i], "contents": content, "present_keyphrases": present_keyphrases}
        
        if document_vector_status:
            to_append["vector"] = documents_vectors[i]

        documents.append(to_append)

    return documents


def convert_dataset_to_pyserini_format_kg_index(
        dataset, 
        ids, 
        keyword_for_document_expansion_path, 
        documents_vectors,
        tokenizer,
        length_penalty = -0.25
):
    if keyword_for_document_expansion_path:
        with open(keyword_for_document_expansion_path) as f:
            expansion_keywords = json.load(f)

        assert len(expansion_keywords) == len(dataset)
    else:
        expansion_keywords = [{} for _ in range(len(dataset))]

    documents = []
    for i, text in enumerate(tqdm(dataset)):
        kw_exp = expansion_keywords[i].get("automatically_extracted_keyphrases")
        tokens_scores = Counter(documents_vectors[i])

        if isinstance(kw_exp, dict):
            present_keyphrases = kw_exp.get("present_keyphrases")
        else:
            present_keyphrases = []

        present_keyphrases_tokens = [tokenizer(phrase) for phrase in present_keyphrases]
        present_keyphrases_scores = [sum(tokens_scores[token] for token in tokens) / (len(tokens) - length_penalty) for tokens in present_keyphrases_tokens]
        present_keyphrases_scores = [[kp, score] for kp, score in zip(present_keyphrases, present_keyphrases_scores) if score > 0]
        present_keyphrases_scores = list(sorted(present_keyphrases_scores, key = lambda x: -x[1]))[:10]
        # print(present_keyphrases_scores)
        to_append = {"id": ids[i], 
                     "contents": text, 
                    #  "present_keyphrases_scores": present_keyphrases_scores,
                     "present_keyphrases": [item[0] for item in present_keyphrases_scores],
                     "vector": documents_vectors[i]}
        
        documents.append(to_append)

    return documents