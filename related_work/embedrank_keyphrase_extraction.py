import sent2vec, torch
import numpy as np
import torch.nn.functional as F
from erukg.nounphrase_extractor import CandidateExtractorRegExpNLTK
from transformers import AutoTokenizer, AutoModel

CANDEXT = CandidateExtractorRegExpNLTK([1,5])
EMBEDDING_MODELS = {

}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"embedrank will be using {DEVICE}")

def vector_normalization(vectors):
    normalized_vector = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return normalized_vector

def embed_sentences_sent2vec(sentences):
    # init model if not already initiated
    if not EMBEDDING_MODELS.get("sent2vec"):
        print("Init model, this will not be done a second time")
        model = sent2vec.Sent2vecModel()
        model.load_model("/scratch/lamdo/wiki_unigrams.bin")
        EMBEDDING_MODELS["sent2vec"] = model
    
    res = EMBEDDING_MODELS["sent2vec"].embed_sentences(sentences)

    return np.array(res)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_sentences_sentence_transformer(sentences, model_name = 'sentence-transformers/all-MiniLM-L6-v2'):
    if not EMBEDDING_MODELS.get("sentence_transformers"):
        print(f"Init SBERT model ({model_name}), this will not be done a second time")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(DEVICE)

        EMBEDDING_MODELS["sentence_transformers"] = {}
        EMBEDDING_MODELS["sentence_transformers"]["tokenizer"] = tokenizer
        EMBEDDING_MODELS["sentence_transformers"]["model"] = model

    encoded_input = EMBEDDING_MODELS["sentence_transformers"]["tokenizer"](sentences, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

    # Compute token embeddings
    with torch.no_grad():
        model_output = EMBEDDING_MODELS["sentence_transformers"]["model"](**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings.cpu().numpy()
        


def embedrank_keyphrase_extraction(doc, embed_func = embed_sentences_sent2vec, top_k = 10):
    if not doc: return []
    candidates = CANDEXT(doc)
    if not candidates: return []

    candidates_embeddings = vector_normalization(embed_func(candidates))
    document_embedding = vector_normalization(embed_func([doc.lower()]))

    scores = document_embedding.dot(candidates_embeddings.T).reshape(-1)

    assert len(scores) == len(candidates)
    candidates_scores = [[candidates[i], scores[i]] for i in range(len(scores))]

    candidates_scores = list(sorted(candidates_scores, key = lambda x: -x[1]))

    return {
        "present": candidates_scores[:top_k],
        "absent": []
    }