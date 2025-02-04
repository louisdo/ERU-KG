import re
import os
import json
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from nltk import PorterStemmer
from sys import exit

pd.options.mode.chained_assignment = None

from src.promptrank_helper.data import data_process_custom

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS = {}

MAX_LEN = None
enable_pos = None
temp_en = None
temp_de = None
length_factor = None
position_factor = None
tokenizer = None


def init(setting_dict):
    '''
    Init template, max length and tokenizer.
    '''

    global MAX_LEN, temp_en, temp_de, tokenizer
    global enable_pos, length_factor, position_factor

    MAX_LEN = setting_dict["max_len"]
    temp_en = setting_dict["temp_en"]
    temp_de = setting_dict["temp_de"]
    enable_pos = setting_dict["enable_pos"]
    position_factor = setting_dict["position_factor"]
    length_factor = setting_dict["length_factor"]


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


def init_promptrank(model_run_index):
    global MODELS
    if MODELS.get("model_run_index") != model_run_index:
        setting_dict = get_setting_dict()
        init(setting_dict)
        model = T5ForConditionalGeneration.from_pretrained("t5-"+ setting_dict["model"])
        tokenizer = T5Tokenizer.from_pretrained("t5-" + setting_dict["model"], model_max_length=MAX_LEN)

        model.to(DEVICE)

        MODELS = {
            "model_run_index": model_run_index,
            "model": model,
            "tokenizer": tokenizer,
            "setting_dict": setting_dict
            }


def promptrank_keyphrase_generation(doc_list, dataloader, topk = 50, model_run_index = 1):
    init_promptrank(model_run_index)

    res = generate_keyphrases(MODELS['model'], MODELS['tokenizer'], doc_list, dataloader, topk)

    # ([en_input_ids, en_input_mask, de_input_ids, dic], doc) = doc
    # res = generate_keyphrases_single_doc(MODELS['model'], MODELS['tokenizer'], doc, en_input_ids, en_input_mask, de_input_ids, dic, topk)
    return res


def generate_keyphrases(model, tokenizer, doc_list, dataloader, topk):
    model.eval()

    cos_similarity_list = {}
    candidate_list = []
    cos_score_list = []
    doc_id_list = []
    pos_list = []
    
    template_len = tokenizer(temp_de, return_tensors="pt")["input_ids"].shape[1] - 3 # single space
    # print(template_len)

    for id, [en_input_ids,  en_input_mask, de_input_ids, dic] in enumerate(tqdm(dataloader,desc="Evaluating with promptrank: ")):

        en_input_ids = en_input_ids.to(DEVICE)
        en_input_mask = en_input_mask.to(DEVICE)
        de_input_ids = de_input_ids.to(DEVICE)

        score = np.zeros(en_input_ids.shape[0])
        
        # print(dic["candidate"])
        # print(dic["de_input_len"])
        # exit(0)

        with torch.no_grad():
            output = model(input_ids=en_input_ids, attention_mask=en_input_mask, decoder_input_ids=de_input_ids)[0]
            #print(en_output.shape)
            # x = empty_ids.repeat(en_input_ids.shape[0], 1, 1).to(DEVICE)
            # empty_output = model(input_ids=x, decoder_input_ids=de_input_ids)[2]
            
            
            for i in range(template_len, de_input_ids.shape[1] - 3):
                logits = output[:, i, :]
                logits = logits.softmax(dim=1)
                logits = logits.cpu().numpy()

                for j in range(de_input_ids.shape[0]):
                    if i < dic["de_input_len"][j]:
                        score[j] = score[j] + np.log(logits[j, int(de_input_ids[j][i + 1])])
                    elif i == dic["de_input_len"][j]:
                        score[j] = score[j] / np.power(dic["de_input_len"][j] - template_len, length_factor)
                                        # score = score + 0.005 (score - empty_score)
            
               
            doc_id_list.extend(dic["idx"])
            candidate_list.extend(dic["candidate"])
            cos_score_list.extend(score)
            pos_list.extend(dic["pos"])

    cos_similarity_list["doc_id"] = doc_id_list
    cos_similarity_list["candidate"] = candidate_list
    cos_similarity_list["score"] = cos_score_list
    cos_similarity_list["pos"] = pos_list
    cosine_similarity_rank = pd.DataFrame(cos_similarity_list)

    for i in range(len(doc_list)):
        
        doc_len = len(doc_list[i].split())
        
        doc_results = cosine_similarity_rank.loc[cosine_similarity_rank['doc_id']==i]
        if enable_pos == True:
            #doc_results.loc[:,"pos"] = torch.Tensor(doc_results["pos"].values.astype(float)) / doc_len + position_factor / (doc_len ** 3)
            doc_results["pos"] = doc_results["pos"] / doc_len + position_factor / (doc_len ** 3)
            doc_results["score"] = doc_results["pos"] * doc_results["score"]
        #* doc_results["score"].values.astype(float)
        ranked_keyphrases = doc_results.sort_values(by='score', ascending=False)
        top_k = ranked_keyphrases.reset_index(drop = True)
        top_k_can = top_k.loc[:, ['candidate']].values.tolist()
        #print(top_k)
        #exit()
        candidates_set = set()
        candidates_dedup = []
        for temp in top_k_can:
            temp = temp[0].lower()
            if temp in candidates_set:
                continue
            else:
                candidates_set.add(temp)
                candidates_dedup.append(temp)

    res = {
        "present": candidates_dedup[:topk],
        "absent": []
    }

    return res






def generate_keyphrases_single_doc(model, tokenizer, doc, en_input_ids, en_input_mask, de_input_ids, dic, topk):
    model.eval()
    template_len = tokenizer(temp_de, return_tensors="pt")["input_ids"].shape[1] - 3

    if not isinstance(en_input_ids, torch.Tensor):
        en_input_ids = torch.tensor(en_input_ids)
    if not isinstance(en_input_mask, torch.Tensor):
        en_input_mask = torch.tensor(en_input_mask)

    en_input_ids = en_input_ids.unsqueeze(0)
    en_input_mask = en_input_mask.unsqueeze(0)
    de_input_ids = de_input_ids.unsqueeze(0)
    en_input_ids = en_input_ids.to(DEVICE)
    en_input_mask = en_input_mask.to(DEVICE)
    de_input_ids = de_input_ids.to(DEVICE)

    score = np.zeros(de_input_ids.shape[0])

    with torch.no_grad():
        output = model(
            input_ids=en_input_ids, 
            attention_mask=en_input_mask, 
            decoder_input_ids=de_input_ids
        )[0]

        for i in range(template_len, de_input_ids.shape[1] - 3):
            logits = output[:, i, :].softmax(dim=1)
            logits = logits.cpu().numpy()
            for j in range(de_input_ids.shape[0]):
                if i < dic["de_input_len"][j]:
                    next_token_id = int(de_input_ids[j][i + 1])
                    score[j] += np.log(logits[j, next_token_id])
                elif i == dic["de_input_len"][j]:
                    denom = (dic["de_input_len"][j] - template_len) ** length_factor
                    score[j] /= denom

    keyphrases_df = pd.DataFrame({
        "candidate": dic["candidate"],
        "score": score
    })
    if "pos" in dic:
        keyphrases_df["pos"] = dic["pos"]
    else:
        keyphrases_df["pos"] = 0.0

    if enable_pos:
        doc_len = len(doc.split())
        keyphrases_df["pos"] = (
            keyphrases_df["pos"] / doc_len + position_factor / (doc_len ** 3)
        )
        keyphrases_df["score"] = keyphrases_df["pos"] * keyphrases_df["score"]

    ranked_keyphrases = keyphrases_df.sort_values(by='score', ascending=False)
    ranked_keyphrases = ranked_keyphrases.reset_index(drop=True)

    seen = set()
    deduped = []
    for phrase in ranked_keyphrases["candidate"]:
        lower_p = phrase.lower()
        if lower_p not in seen:
            seen.add(lower_p)
            deduped.append(lower_p)

    # Return top-k
    res = {
        "present": deduped[:topk],
        "absent" : []
    }
    
    return res


