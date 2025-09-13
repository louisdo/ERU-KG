import torch
import numpy as np
import torch.nn.functional as F
from transformers import EncoderDecoderModel, BertTokenizer
from erukg.two_stage_keyphrase_extraction_with_splade import init_splade_model, SPLADE_MODEL, get_tokens_scores_of_doc
from transformers import LogitsProcessor
from collections import Counter

PHRASENESS_MODEL_NAME = "phraseness_500k_v2"
INFORMATIVENESS_MODEL_NAME = ""

PHRASENESS_MODEL = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BiasLogitsProcessor(LogitsProcessor):
    def __init__(self, bias_weights, input_ids):
        self.bias_weights = bias_weights

    def __call__(self, input_ids, scores):
        # Ensure the bias_weights are on the same device as scores
        beam_size = input_ids.shape[0]
        bias_weights = self.bias_weights.to(scores.device).repeat(beam_size, 1)

        # print(scores.shape, scores)

        # print(bias_weights.shape)

        if input_ids.shape[1] > 1:
            input_ids_without_bos = input_ids[:,1:]
            current_generation_scores = torch.tensor([torch.mean(self.bias_weights[0][indices]) for indices in input_ids_without_bos]).to(scores.device)
            # print(current_generation_scores.shape)

            bias_weights[:, 102] = current_generation_scores

        # bias_weights = torch.exp(bias_weights)
        normalization = torch.sum(bias_weights, dim = 1).reshape(-1, 1)
        # print(normalization.shape)
        bias_weights /= normalization
        bias_weights = torch.log(bias_weights)

        # print(torch.max(scores), torch.mean(bias_weights))

        assert scores.shape == bias_weights.shape, [scores.shape, bias_weights.shape]
        
        res = 0.7 * scores + bias_weights

        return res

def init_phraseness_module(model_name = PHRASENESS_MODEL_NAME):

    if PHRASENESS_MODEL.get(model_name) is not None:
        return 
    
    model_type_or_dir = None

    if model_name == "phraseness_200k":
        model_type_or_dir = "/scratch/lamdo/phraseness_module/checkpoints/test_"
    elif model_name == "phraseness_500k":
        model_type_or_dir = "/scratch/lamdo/phraseness_module/checkpoints/phraseness_module_500k"
    elif model_name == "phraseness_500k_v2":
        model_type_or_dir = "/scratch/lamdo/phraseness_module/checkpoints/phraseness_module_500k_v2"

    model = EncoderDecoderModel.from_pretrained(model_type_or_dir).to(DEVICE)
    tokenizer = BertTokenizer.from_pretrained(model_type_or_dir)

    PHRASENESS_MODEL[model_name] = {
        "model": model,
        "tokenizer": tokenizer
    }

    return True


def init_informativeness_module(model_name):
    init_splade_model(model_name)



def score_candidates_by_positions(candidates, doc):
    res = Counter()
    for cand in candidates:
        try:
            temp = doc.index(cand)
            position = len([item for item in doc[:temp].split(" ") if item]) + 1
            position_score =  position / (position + 1)
        except ValueError:
            position_score = 1
        res[cand] = position_score
    return res


def generate_keyphrases(doc, 
                        top_k = 10,
                        phraseness_model_name = PHRASENESS_MODEL_NAME,
                        informativeness_model_name = INFORMATIVENESS_MODEL_NAME,
                        apply_position_penalty = False,
                        length_penalty = 0,
                        return_only_present = True):
    doc = doc.lower()
    init_informativeness_module(informativeness_model_name)
    init_phraseness_module(phraseness_model_name)
    informativeness_term_scores = SPLADE_MODEL[informativeness_model_name]["model"](
        d_kwargs=SPLADE_MODEL[informativeness_model_name]["tokenizer"](doc, return_tensors="pt", max_length = 512, truncation = True).to(DEVICE))["d_rep"].squeeze().reshape(1, -1)


    # print(get_tokens_scores_of_doc(doc, model_name=informativeness_model_name))

    # generate keyphrases
    inputs = PHRASENESS_MODEL[phraseness_model_name]["tokenizer"](doc, return_tensors="pt").to(DEVICE)
    logits_processor = BiasLogitsProcessor(informativeness_term_scores, inputs.input_ids)
    output = PHRASENESS_MODEL[phraseness_model_name]["model"].generate(
        **inputs,
        max_new_tokens = 10,
        logits_processor=[logits_processor],
        num_beams=100,
        num_return_sequences=100,
        no_repeat_ngram_size=1,
        temperature = 0.7,
        output_scores=True,
        return_dict_in_generate=True,
        # diversity_penalty = 1.1,
        # num_beam_groups = 5,
        # do_sample = True
    )

    sequence_lengths = (output.sequences[:,1:] != 0).sum(dim=1)

    # print(output)
    scores =  output.sequences_scores * sequence_lengths / (sequence_lengths + length_penalty)
    sequences = PHRASENESS_MODEL[phraseness_model_name]["tokenizer"].batch_decode(output.sequences, skip_special_tokens=True)
    res = [[seq.replace(" - ", "-"), score] for seq, score in zip(sequences, scores)]

    if apply_position_penalty:
        position_scores = score_candidates_by_positions(candidates = [item[0] for item in res], doc = doc)
        res = [[item[0], item[1] * position_scores[item[0]]] for item in res]

    res = list(sorted(res, key = lambda x: -x[1]))

    # this is temporary code, filtering the phrases that exist in documents
    if return_only_present:
        res = [item for item in res if item[0] in doc]
    dedup_res = []
    visited = set([])
    for item in res:
        if item[0] not in visited:
            dedup_res.append(item)
            visited.add(item[0])

    return dedup_res[:top_k]


if __name__ == "__main__":
    text = """With the rapid increase in paper submissions to academic conferences, the need for automated and accurate paper-reviewer matching is more critical than ever. Previous efforts in this area have
considered various factors to assess the relevance of a reviewer’s
expertise to a paper, such as the semantic similarity, shared topics,
and citation connections between the paper and the reviewer’s
previous works. However, most of these studies focus on only one
factor, resulting in an incomplete evaluation of the paper-reviewer
relevance. To address this issue, we propose a unified model for
paper-reviewer matching that jointly considers semantic, topic, and
citation factors. To be specific, during training, we instruction-tune
a contextualized language model shared across all factors to capture their commonalities and characteristics; during inference, we
chain the three factors to enable step-by-step, coarse-to-fine search
for qualified reviewers given a submission. Experiments on four
datasets (one of which is newly contributed by us) spanning various fields such as machine learning, computer vision, information
retrieval, and data mining consistently demonstrate the effectiveness of our proposed Chain-of-Factors model in comparison with
state-of-the-art paper-reviewer matching methods and scientific
pre-trained language models""".lower()
    
    test = generate_keyphrases(doc = text, top_k = 10,
                               informativeness_model_name="custom_trained_combined_references_v8")
    
    print(test)