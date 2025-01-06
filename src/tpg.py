import re
import os
import json
import torch
import argparse
from nltk.stem import PorterStemmer
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

#nltk.download('stopwords')

from transformers import BartForConditionalGeneration, BartTokenizer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS = {}

def generate_keyphrases(model, text, tokenizer, device, max_length=512, num_beams=20):
    
    model.eval()  
    model.to(device)  

    with torch.no_grad():
        inputs = tokenizer(text, max_length=512, truncation=True, padding='longest', return_tensors="pt").to(device) 
        outputs = model.generate(inputs['input_ids'], num_beams=num_beams, max_length=max_length, num_return_sequences=num_beams)
        #output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True) 

    return output_text


def extract_from_beams(seq_list):
    total_phrases = []
    for seq in seq_list:
        phrases = seq.split(';')
        phrases = [ p.strip() for p in phrases if p.strip() != '']
        for phrase in phrases:
            if phrase not in total_phrases:
                total_phrases.append(phrase)
    return total_phrases


def init_tpg(model_run_index):
    global MODELS, DEVICE
    if MODELS.get("model_run_index") != model_run_index:
        print(f"Initializing model TPG #{model_run_index}. This will be done only once")
        model_path = f"/scratch/lamdo/tpg_checkpoints/tpg_bart_trained_{model_run_index}/"

        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        special_tokens_dict = {'additional_special_tokens': ['[sep]', '[digit]']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        model = BartForConditionalGeneration.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))

        device = DEVICE
        model.to(device)

        MODELS = {
            "model_run_index": model_run_index,
            "model": model,
            "tokenizer": tokenizer
        }


def tpg_keyphrase_generation(doc, top_k, model_run_index):
    init_tpg(model_run_index)


    extracted_phrases_seq = generate_keyphrases(MODELS["model"], doc, MODELS["tokenizer"], DEVICE, num_beams=top_k)
    extracted_phrases = extract_from_beams(extracted_phrases_seq)

    present_keyphrases = [[kp, None] for kp in extracted_phrases if kp in doc]
    absent_keyphrases = [[kp, None] for kp in extracted_phrases if kp not in doc]

    res = {
        "present": present_keyphrases[:top_k],
        "absent": absent_keyphrases[:top_k]
    }

    return res