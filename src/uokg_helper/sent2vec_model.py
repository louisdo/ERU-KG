import nltk, string, json
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
STOPWORDS = nltk.corpus.stopwords.words("english")
from typing import Iterable, List
import numpy as np

class Sent2VecModel:
    def __init__(self, config):
        with open(config["vocab_path"]) as f:
            self.vocab = json.load(f)
            self.word2index = {self.vocab[i]:i for i in range(len(self.vocab))}
        
        self.uni_embs = np.load(config["uni_embs_path"])
        self.EMB_SIZE = 600
        self.punctuation = set(string.punctuation)

    def tokenize(self, sentence: str):
        _res = nltk.tokenize.word_tokenize(sentence)
        return [tok for tok in _res if tok not in self.punctuation]

    def embed_sentence(self, sentence, pooling = "mean", drop_stopwords = False):
        tokens = self.tokenize(sentence)
        if drop_stopwords == True:
            tokens = [tok for tok in tokens if tok not in STOPWORDS]
        indices = [self.word2index[tok] for tok in tokens if tok in self.word2index]
        if len(indices) == 0: return np.zeros((1,self.EMB_SIZE))

        token_embeddings = self.uni_embs[indices]
        if pooling == "mean":
            sentence_embedding = np.sum(token_embeddings, axis = 0).reshape(1, -1) / len(tokens)
        elif pooling == "sum":
            sentence_embedding = np.sum(token_embeddings, axis = 0).reshape(1, -1)
        elif pooling == "max":
            sentence_embedding = np.max(token_embeddings, axis = 0).reshape(1, -1)
        else:
            raise NotImplementedError(f"'{pooling}' pooling method is not implemented")
        return sentence_embedding # [1, emb_size]

    def embed_sentences(self, sentences: list, pooling: str = "mean", drop_stopwords = False):
        sentences_embeddings = [self.embed_sentence(sentence, pooling = pooling, drop_stopwords = drop_stopwords) for sentence in sentences]
        return np.concatenate(sentences_embeddings, axis = 0) # [num_sentences, emb_size]

    
    def embed_tokenized_sentence(self, tokenized_sentence: List[str], pooling = "mean"):
        indices = [self.word2index[tok] for tok in tokenized_sentence if tok in self.word2index]
        if len(indices) == 0: return np.zeros((1,self.EMB_SIZE))

        token_embeddings = self.uni_embs[indices]
        if pooling == "mean":
            sentence_embedding = np.sum(token_embeddings, axis = 0).reshape(1, -1) / len(tokenized_sentence)
        elif pooling == "sum":
            sentence_embedding = np.sum(token_embeddings, axis = 0).reshape(1, -1)
        elif pooling == "max":
            sentence_embedding = np.max(token_embeddings, axis = 0).reshape(1, -1)
        else:
            raise NotImplementedError(f"'{pooling}' pooling method is not implemented")
        return sentence_embedding # [1, emb_size]

    def embed_tokenized_sentences(self, tokenized_sentences: List[List[str]], pooling = "mean"):
        sentences_embeddings = [self.embed_tokenized_sentence(tokenized_sentence, pooling = pooling) \
        for tokenized_sentence in tokenized_sentences]
        return np.concatenate(sentences_embeddings, axis = 0) # [num_sentences, emb_size]


    def embed_words(self, word_list):
        return self.embed_sentences(word_list) #[num_words, emb_size]
        # res = [self.uni_embs[self.word2index[word]].reshape(1, -1) if word in self.word2index \
        # else np.zeros((1,self.EMB_SIZE)) for word in word_list]
        # return np.concatenate(res, axis = 0) #[num_words, emb_size]