#coding=utf-8
import re
import codecs
import json
import os

from torch.utils.data import Dataset
from transformers import T5Tokenizer
import pandas as pd
import nltk
from nltk.corpus import stopwords

from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

MAX_LEN = None
enable_filter = None
temp_en = None
temp_de = None

StanfordCoreNLP_path = '/home/abodke2/kpg/stanford-corenlp-full-2018-02-27'

stopword_dict = set(stopwords.words('english'))
en_model = StanfordCoreNLP(StanfordCoreNLP_path, quiet=True)
# tokenizer = None




def get_setting_dict():
    setting_dict = {}
    setting_dict["max_len"] = 512
    setting_dict["temp_en"] = "Book:"
    setting_dict["temp_de"] = "This book mainly talks about "
    setting_dict["model"] = "small"
    setting_dict["enable_filter"] = False
    setting_dict["enable_pos"] = True
    setting_dict["position_factor"] = 1.2e8
    setting_dict["length_factor"] = 0.6
    return setting_dict

setting_dict = get_setting_dict()
tokenizer = T5Tokenizer.from_pretrained("t5-" + setting_dict["model"], model_max_length=setting_dict["max_len"] )

GRAMMAR = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

def extract_candidates(tokens_tagged, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """

    cans_count = dict()
    
    np_parser = nltk.RegexpParser(GRAMMAR)  # Noun phrase parser
    keyphrase_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ' '.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            
            if len(np.split()) == 1:
                if np not in cans_count.keys():
                    cans_count[np] = 0
                cans_count[np] += 1
                
            keyphrase_candidate.append((np, start_end))

        else:
            count += 1
        
    # if enable_filter == True:
    #     i = 0
    #     while i < len(keyphrase_candidate):
    #         can, pos = keyphrase_candidate[i]
    #         #pos[0] > 50 and
    #         if can in cans_count.keys() and cans_count[can] == 1:
    #             keyphrase_candidate.pop(i)
    #             continue
    #         i += 1
    
    return keyphrase_candidate

class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, en_model, text=""):
        """
        :param is_sectioned: If we want to section the text.
        :param en_model: the pipeline of tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}

        self.tokens = []
        self.tokens_tagged = []
        self.tokens = en_model.word_tokenize(text)
        self.tokens_tagged = en_model.pos_tag(text)
        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):
            if token.lower() in stopword_dict:
                self.tokens_tagged[i] = (token, "IN")
        self.keyphrase_candidate = extract_candidates(self.tokens_tagged, en_model)
        
class KPE_Dataset(Dataset):
    def __init__(self, docs_pairs):

        self.docs_pairs = docs_pairs
        self.total_examples = len(self.docs_pairs)

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):

        doc_pair = self.docs_pairs[idx]
        en_input_ids = doc_pair[0][0]
        en_input_mask = doc_pair[1][0]
        de_input_ids = doc_pair[2][0]
        dic = doc_pair[3]

        return [en_input_ids, en_input_mask, de_input_ids, dic]
    
def clean_text(text="",database="Inspec"):

    #Specially for Duc2001 Database
    if(database=="Duc2001" or database=="Semeval2017"):
        pattern2 = re.compile(r'[\s,]' + '[\n]{1}')
        while (True):
            if (pattern2.search(text) is not None):
                position = pattern2.search(text)
                start = position.start()
                end = position.end()
                # start = int(position[0])
                text_new = text[:start] + "\n" + text[start + 2:]
                text = text_new
            else:
                break

    pattern2 = re.compile(r'[a-zA-Z0-9,\s]' + '[\n]{1}')
    while (True):
        if (pattern2.search(text) is not None):
            position = pattern2.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + " " + text[start + 2:]
            text = text_new
        else:
            break

    pattern3 = re.compile(r'\s{2,}')
    while (True):
        if (pattern3.search(text) is not None):
            position = pattern3.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + "" + text[start + 2:]
            text = text_new
        else:
            break

    pattern1 = re.compile(r'[<>[\]{}]')
    text = pattern1.sub(' ', text)
    text = text.replace("\t", " ")
    text = text.replace(' p ','\n')
    text = text.replace(' /p \n','\n')
    lines = text.splitlines()
    # delete blank line
    text_new=""
    for line in lines:
        if(line!='\n'):
            text_new+=line+'\n'

    return text_new

def remove (text):
    text_len = len(text.split())
    remove_chars = '[’!"#$%&\'()*+,./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, '', text)
    re_text_len = len(text.split())
    if text_len != re_text_len:
        return True
    else:
        return False
    
def generate_doc_pairs(doc, candidates):
    count = 0
    doc_pairs = []
    
    en_input =  tokenizer(doc, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    en_input_ids = en_input["input_ids"]
    en_input_mask = en_input["attention_mask"]
    
    for id, can_and_pos in enumerate(candidates):
        candidate = can_and_pos[0]
        # Remove stopwords in a candidate
        if remove(candidate):
            count +=1
            continue
    
        de_input = temp_de + candidate + " ."
        de_input_ids = tokenizer(de_input, max_length=30, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
        de_input_ids[0, 0] = 0
        de_input_len = [(de_input_ids[0] == tokenizer.eos_token_id).nonzero()[0].item() - 2]
        
#         for i in de_input_ids[0]:
#             print(tokenizer.decode(i))
#         print(de_input_len)
        
        
#         x = tokenizer(temp_de, return_tensors="pt")["input_ids"]
#         for i in x[0]:
#             print(tokenizer.decode(i))
#         exit(0)
        dic = {"de_input_len":de_input_len, "candidate":candidate, 'idx' : 0, "pos":can_and_pos[1][0]}
        
        doc_pairs.append([en_input_ids, en_input_mask, de_input_ids, dic])
        # print(tokenizer.decode(en_input_ids[0]))
        # print(tokenizer.decode(de_input_ids[0]))
        # print(candidate)
        # print(de_input_len)
        # print()
        # exit(0)
    return doc_pairs, count


def init(setting_dict):
    '''
    Init template, max length and tokenizer.
    '''

    global MAX_LEN, temp_en, temp_de, tokenizer, enable_filter
    MAX_LEN = setting_dict["max_len"]
    temp_en = setting_dict["temp_en"]
    temp_de = setting_dict["temp_de"]
    enable_filter = setting_dict["enable_filter"]

    # tokenizer = T5Tokenizer.from_pretrained("t5-" + setting_dict["model"], model_max_length=MAX_LEN)

def process_single_doc(doc, setting_dict):
    init(setting_dict)
    text_obj = InputTextObj(en_model, doc)
    
    candidates = []
    for candidate, pos in text_obj.keyphrase_candidate:
        if enable_filter and len(candidate.split()) > 4:
            continue
        candidates.append([candidate.lower(), pos])
    
    formatted_doc = temp_en + "\"" + doc + "\""
    
    doc_pairs, removed_count = generate_doc_pairs(formatted_doc, candidates)
    dataset = KPE_Dataset(doc_pairs)
    # en_model.close()

    return dataset



def data_process_custom(setting_dict, dataset_name, size = 'large'):
    init(setting_dict)
    df = pd.read_json(f"hf://datasets/memray/{dataset_name}/test.json", lines=True)
    data = {}
    labels = {}
    # print(df.shape)
    for idx, line in enumerate(df.to_dict("records")):
        if size == 'small' and idx == 1:
            break 
        keywords = line.get("keywords").lower().split(";")
        title = line.get("title")
        abstract = line.get("abstract")

        # fulltxt = jsonl['fulltext']
        doc = f"{title}\n{abstract}"
        doc = re.sub('\. ', ' . ', doc)
        doc = re.sub(', ', ' , ', doc)

        doc = clean_text(doc, database="nus")
        doc = doc.replace('\n', ' ')
        data[line.get('name')] = doc
        # labels[line.get('name')] = keywords
   
    docs_pairs = []
    doc_list = []
    labels = []
    labels_stemed = []
    t_n = 0
    candidate_num = 0
    porter = nltk.PorterStemmer()

    for idx, (key, doc) in enumerate(data.items()):

        # Get stemmed labels and document segments
        # labels.append([ref.replace(" \n", "") for ref in labels[key]])
        # labels_s = []
        # for l in labels[key]:
        #     tokens = l.split()
        #     labels_s.append(' '.join(porter.stem(t) for t in tokens))

        doc = ' '.join(doc.split()[:MAX_LEN])  
        # labels_stemed.append(labels_s)
        doc_list.append(doc)
        
        # Statistic on empty docs
        empty_doc = 0
        try:
            text_obj = InputTextObj(en_model, doc)
        except:
            empty_doc += 1
            print("doc: ", doc)

        # Generate candidates (lower)
        cans = text_obj.keyphrase_candidate
        candidates = []
        for can, pos in cans:
            # if enable_filter == True and len(can.split()) > 4:
            #     continue
            candidates.append([can.lower(), pos])
        candidate_num += len(candidates)
        
        # Generate docs_paris for constructing dataset 
        # doc = doc.lower()
        doc = temp_en + "\"" + doc + "\""
        doc_pairs, count = generate_doc_pairs(doc, candidates, idx)
        docs_pairs.extend(doc_pairs)
        t_n += count
    

    print("candidate_num: ", candidate_num)
    print("unmatched: ", t_n)
    dataset = KPE_Dataset(docs_pairs)
    print("examples: ", dataset.total_examples)

    # en_model.close()
    print(len(dataset), len(doc_list))
    return dataset, doc_list
