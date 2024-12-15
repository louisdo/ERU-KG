import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = {"model": None, "tokenizer": None}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "taln-ls2n/bart-base-kp20k" # "bloomberg/KeyBART"

print(f"KeyBART will be using {DEVICE}")

def is_sublist(sublist, main_list):
    if len(sublist) > len(main_list):
        return False
    
    return any(sublist == main_list[i:i+len(sublist)] for i in range(len(main_list) - len(sublist) + 1))

def init_model():
    if MODEL["model"]: return True
    print("Initiating model, this will be done only once")
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    model = model.to(DEVICE)

    MODEL["model"] = model
    MODEL["tokenizer"] = tokenizer

    return True


def generate_keywords(text, 
                      top_k,
                      max_length=128, 
                      num_return_sequences=1):
    
    init_model()
    # Tokenize the input text
    inputs = MODEL["tokenizer"](text, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
    
    # Generate keywords
    outputs = MODEL["model"].generate(
        **inputs
        # max_length=max_length,
        # num_return_sequences=num_return_sequences,
        # num_beams=num_return_sequences,
        # no_repeat_ngram_size=3
    ).cpu()
    
    # Decode the generated outputs
    # print(outputs[0])
    keywords = [MODEL["tokenizer"].decode(output, skip_special_tokens=True) for output in outputs][0]
    
    res = [kw.strip().lower() for kw in keywords.split(";")]
    res = [kw for kw in res if kw]

    keywords_tokens = [MODEL["tokenizer"](kw).input_ids[1:-1] for kw in res]

    inputs_tokens = inputs.input_ids.tolist()

    present_indices = [i for i in range(len(keywords_tokens)) if is_sublist(keywords_tokens[i], inputs_tokens[0])]
    absent_indices = [i for i in range(len(keywords_tokens)) if i not in present_indices]

    present_keyphrases = [res[i] for i in present_indices]
    absent_keyphrases = [res[i] for i in absent_indices]

    return {
        "present":present_keyphrases[:top_k],
        "absent": absent_keyphrases[:top_k]
    }



if __name__=="__main__":

    # Example usage
    input_text = """2-source dispersers for sub-polynomial entropy and ramsey graphs beating the frankl-wilson construction\nthe main result of this paper is an explicit disperser for two independent sources on n bits, each of entropy k = n o(1). put differently, setting n = 2n and k = 2k , we construct explicit n  n boolean matrices for which no k  k sub-matrix is monochromatic. viewed as adjacency matrices of bipartite graphs, this gives an explicit construction of k-ramsey bipartite graphs of size n . this greatly improves the previous bound of k = o(n) of barak, kindler, shaltiel, sudakov and wigderson [4]. it also significantly improves the 25-year record of k = ~ o(n) on the special case of ramsey graphs, due to frankl and wilson [9]. the construction uses (besides \"classical\" extractor ideas) almost all of the machinery developed in the last couple of years for extraction from independent sources, including: bourgain's extractor for 2 independent sources of some entropy rate &lt; 1/2 [5] raz's extractor for 2 independent sources, one of which has any entropy rate &gt; 1/2 [18] rao's extractor for 2 independent block-sources of entropy n (1) [17] the \"challenge-response\" mechanism for detecting \"entropy concentration\" of [4]. the main novelty comes in a bootstrap procedure which allows the challenge-response mechanism of [4] to be used with sources of less and less entropy, using recursive calls to itself. subtleties arise since the success of this mechanism depends on restricting the given sources, and so recursion constantly changes the original sources. these are resolved via a new construct, in between a disperser and an extractor, which behaves like an extractor on sufficiently large subsources of the given ones. this version is only an extended abstract, please see the full version, available on the authors' homepages, for more details."""

    generated_keywords = generate_keywords(input_text, top_k = 10)

    print(generated_keywords)