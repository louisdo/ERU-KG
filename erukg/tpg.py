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
        phrases = [ p.strip().lower() for p in phrases if p.strip() != '']
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

    lower_doc = doc.lower()


    extracted_phrases_seq = generate_keyphrases(MODELS["model"], lower_doc, MODELS["tokenizer"], DEVICE, num_beams=20)
    extracted_phrases = extract_from_beams(extracted_phrases_seq)

    present_keyphrases = [[kp, None] for kp in extracted_phrases if kp in lower_doc]
    absent_keyphrases = [[kp, None] for kp in extracted_phrases if kp not in lower_doc]

    res = {
        "present": present_keyphrases[:top_k],
        "absent": absent_keyphrases[:top_k]
    }

    return res


if __name__ == "__main__":
    index = 3
    test = tpg_keyphrase_generation("Real World BCI: Cross-Domain Learning and Practical Applications", 10, index)
#     test = tpg_keyphrase_generation("""I am seeking alternatives to Generative Adversarial Networks (GANs) that can be applied to image datasets, such as CIFAR-10. The alternative
# should be capable of generating new data points based on the original data distribution and should perform comparably to GANs across various
# metrics. Could you provide information on the standard metrics typically used to evaluate the performance of GANs? I anticipate that this alternative
# method would initially estimate and model the original data distribution, possibly using a neural network, and then generate diverse data points that
# adhere to the same distribution through an intelligent sampling technique. However, I am open to learning about other promising approaches as well.""", 10, index)
    
#     test = tpg_keyphrase_generation("""Supplementing Remote Sensing of Ice: Deep Learning-Based Image Segmentation System for Automatic Detection and Localization of Sea-ice Formations From Close-Range Optical Images
# This paper presents a three-stage approach for the automated analysis of close-range optical images containing ice objects. The proposed system is based on an ensemble of deep learning models and conditional random field postprocessing. The following surface ice formations were considered: Icebergs, Deformed ice, Level ice, Broken ice, Ice floes, Floebergs, Floebits, Pancake ice, and Brash ice. Additionally, five non-surface ice categories were considered: Sky, Open water, Shore, Underwater ice, and Melt ponds. To find input parameters for the approach, the performance of 12 different neural network architectures was explored and evaluated using a 5-fold cross-validation scheme. The best performance was achieved using an ensemble of models having pyramid pooling layers (PSPNet, PSPDenseNet, DeepLabV3+, and UPerNet) and convolutional conditional random field postprocessing with a mean intersection over union score of 0.799, and this outperformed the best single-model approach. The results of this study show that when per-class performance was considered, the Sky was the easiest class to predict, followed by Deformed ice and Open water. Melt pond was the most challenging class to predict. Furthermore, we have extensively explored the strengths and weaknesses of our approach and, in the process, discovered the types of scenes that pose a more significant challenge to the underlying neural networks. When coupled with optical sensors and AIS, the proposed approach can serve as a supplementary source of large-scale `ground truth' data for validation of satellite-based sea-ice products. We have provided an implementation of the approach at https://github.com/panchinabil/sea_ice_segmentation.""", 10, index)

    present_keyphrases = ", ".join([item[0] for item in test["present"]])
    absent_keyphrases = ", ".join([item[0] for item in test["absent"]])

    with open("test_gitig_.txt", "a") as f:
        to_write = f"""present: {present_keyphrases}
absent: {absent_keyphrases}\n\n\n"""
        f.write(to_write)