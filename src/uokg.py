import sys, nltk, torch, math, os, json
import numpy as np
from src.uokg_helper.sent2vec_model import Sent2VecModel
from collections import Counter
from src.uokg_helper.models.PhraseLM_PG_POS_Version2 import TransformerPointerGenerator
from src.uokg_helper.phrase_retrieval.build_index import load_index
from src.uokg_helper.phrase_retrieval.phrase_retriever import knn_search, knn_search_batch, knn_search_return_score, mmr



#------------------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS = {}

LAMB = 0.75
NUM_REFERENCES = 15
RERANK_BEAMSEARCH = True

# RESULT_TEXT_FILE = os.path.join("./temp_v3/temp", os.path.dirname(CKPT_PATH).split("/")[-1] + ".txt")
# print("Eval for model at:", CKPT_PATH)
# print("Results will be written to:", RESULT_TEXT_FILE)

#------------------------------------------------------------------------------------
from collections import Counter
import json, random, pickle, os
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
from tqdm import tqdm


from collections import Counter


# Define special symbols and indices, <unk> symbol will be in the last positions
PAD_IDX, BOS_IDX, EOS_IDX, URL_IDX, EMAIL_IDX, PHONE_IDX, TGT_UNK_IDX, BOR_IDX, EOR_IDX, SEP_IDX = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
# TGT_UNK are for the tokens that neither appear in the vocab nor the text

class Lang:
    def __init__(self, num_position_markers = 1):
        assert num_position_markers >= 1
        self.num_position_markers = num_position_markers


    def build_vocab(self, 
                    data_iter: Iterable,
                    vocab_size: int,
                    mode: str):
        token_counter = Counter()
        for tokens in tqdm(yield_tokens(data_iter, mode = mode)):
            token_counter.update(tokens)
        # Make sure the tokens are in order of their indices to properly insert them in vocab
        special_symbols = ['<pad>', '<bos>', '<eos>', '<url>', '<email>', '<phone>', "<tgt_unk>", "<bor>", "<eor>", "<sep>"]
        self.vocab = special_symbols + list(sorted(token_counter.keys(), key = lambda x: -token_counter[x]))[:vocab_size]

        self.word2index = {self.vocab[index]:index for index in range(len(self.vocab))}
        self.special_symbols = special_symbols

    
    def __len__(self):
        return len(self.vocab) + self.num_position_markers


    def _lookup_index(self, token: str, position = 0):
        # position should start from 0
        assert position is None or position < self.num_position_markers
        if token in self.word2index: return self.word2index[token]
        else: 
            if position is not None:
                return len(self.vocab) + position
            else: return TGT_UNK_IDX


    def lookup_indices(self, tokens: List[str], src_tokens: List[str] = None) -> List[int]:
        assert hasattr(self, "vocab"), "Vocab has not been built"
        if self.num_position_markers == 1:
            # disregard the position of oov token, map to the same index (index of <unk>)
            indices = [self._lookup_index(token) for token in tokens]
        else:
            # regard the position of oov token
            indices = []
            cache = {}
            for i in range(len(tokens)):
                token = tokens[i]
                if token in self.special_symbols:
                    indices.append(self.special_symbols.index(token))
                    continue
                if token not in cache:
                    position = src_tokens.index(token) if ((src_tokens is not None) and (token in src_tokens)) == True else None
                    token_index = self._lookup_index(token, position = position)
                    cache[token] = token_index
                indices.append(cache[token])
        return indices

    def lookup_token(self, index: int, src_tokens: List[int]):
        if index < len(self.vocab): return self.vocab[index]
        else:
            if self.num_position_markers == 1:
                # disregard position of oov token
                return "<unk>"
            else:
                assert index - len(self.vocab) < self.num_position_markers
                if src_tokens is None: 
                    return f"<unk-{index - len(self.vocab)}>"
                else:
                    return src_tokens[index - len(self.vocab)]
    
    
    def lookup_tokens(self, indices: List[int], src_tokens: List[str] = None) -> List[str]:
        assert hasattr(self, "vocab"), "Vocab has not been built"
        return [self.lookup_token(index, src_tokens) for index in indices]


#----------------------------------------------------------------------

from torch.nn.utils.rnn import pad_sequence

def get_log_prob(prob):
    return prob.clamp(1e-10, 1.0).log()

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def transform_text(tokenized_txt_input, pos_tagged_txt_input):
    txt_tokens = tokenized_txt_input[:MODELS["CONFIG"]["num_position_markers"]]
    txt_tokens = MODELS["vocab_transform"]["TEXT"].lookup_indices(txt_tokens)
    txt_tokens = tensor_transform(txt_tokens)
      
    txt_pos_tag_tokens = pos_tagged_txt_input[:MODELS["CONFIG"]["num_position_markers"]]
    txt_pos_tag_tokens = MODELS["vocab_transform"]["POS_TAG"].lookup_indices(txt_pos_tag_tokens)
    txt_pos_tag_tokens = tensor_transform(txt_pos_tag_tokens)
    return txt_tokens, txt_pos_tag_tokens

def transform_phrase(tokenized_phrase_input, pos_tagged_phrase_input, tokenized_txt_input = None):
    phrase_tokens = tokenized_phrase_input[:MODELS["CONFIG"]["num_position_markers"]]
    if tokenized_txt_input:
        phrase_tokens = MODELS["vocab_transform"]["PHRASE"].lookup_indices(
            phrase_tokens, 
            tokenized_txt_input[:MODELS["CONFIG"]["num_position_markers"]])
    else:
        phrase_tokens = MODELS["vocab_transform"]["PHRASE"].lookup_indices(phrase_tokens)
    phrase_tokens = tensor_transform(phrase_tokens)
      
    phrase_pos_tag_tokens = pos_tagged_phrase_input[:MODELS["CONFIG"]["num_position_markers"]]
    phrase_pos_tag_tokens = MODELS["vocab_transform"]["POS_TAG"].lookup_indices(phrase_pos_tag_tokens)
    phrase_pos_tag_tokens = tensor_transform(phrase_pos_tag_tokens)
    return phrase_tokens, phrase_pos_tag_tokens


text_transform = {}
text_transform["TEXT"] = transform_text
text_transform["PHRASE"] = transform_phrase



#------------------------------------------------------------------------------
import time
from sklearn.metrics.pairwise import cosine_similarity as cosine
from sklearn.preprocessing import normalize
class KeynessModel:
    def __init__(self, 
                 sent2vec_model, 
                 vocab_transform,
                 tokenizer,
                 num_position_markers):
        self.sent2vec_model = sent2vec_model
        self.vocab_transform = vocab_transform
        self.tokenizer = lambda x: tokenizer(x)[:num_position_markers]
        self.num_position_markers = num_position_markers

        SENT2VEC_EMB_WORD_VOCAB = sent2vec_model.embed_words(vocab_transform["PHRASE"].vocab)
        SENT2VEC_EMB_WORD_VOCAB = np.concatenate([SENT2VEC_EMB_WORD_VOCAB, 
                                                        np.zeros((num_position_markers, sent2vec_model.EMB_SIZE))], 
                                                        axis = 0) # [vocab_size, emb_size] vocab size = real vocab size + num position markers
        assert SENT2VEC_EMB_WORD_VOCAB.shape == (len(vocab_transform["PHRASE"]), sent2vec_model.EMB_SIZE)
        self.SENT2VEC_EMB_WORD_VOCAB = torch.from_numpy(SENT2VEC_EMB_WORD_VOCAB).to(DEVICE).float()
        


    def reset_emb_phrase_vocab(self):
        VOCAB_LENGTH = len(self.vocab_transform["PHRASE"].vocab)
        self.SENT2VEC_EMB_WORD_VOCAB[VOCAB_LENGTH:, :] = 0


    def fill_emb_phrase_vocab(self, tokenized_document_with_ref):
        self.reset_emb_phrase_vocab()

        document_tokenized = tokenized_document_with_ref
        document_indices = self.vocab_transform["PHRASE"].lookup_indices(document_tokenized, document_tokenized)
        assert len(document_tokenized) == len(document_indices)

        VOCAB_LENGTH = len(self.vocab_transform["PHRASE"].vocab)
        for i in range(len(document_tokenized)):
            tok = document_tokenized[i]
            vocab_index = document_indices[i]
            if vocab_index >= VOCAB_LENGTH:
                self.SENT2VEC_EMB_WORD_VOCAB[vocab_index] = \
                torch.from_numpy(self.sent2vec_model.embed_words([tok]).reshape(-1)).to(DEVICE)

    def get_position_bias(self, document_tokenized):
        # return [1, vocab_size]
        document_indices = self.vocab_transform["PHRASE"].lookup_indices(document_tokenized, document_tokenized)
        position_bias_score = [0.5/math.log(i + 2) for i in range(len(document_tokenized))]
        
        res = np.ones([1, len(self.vocab_transform["PHRASE"])])
        
        assert len(position_bias_score) == len(document_indices)
        visited = set([])
        for index in range(len(position_bias_score)):
            if document_indices[index] in MODELS["vocab_transform"]["PHRASE"].special_symbols or document_indices[index] in visited: continue
            res[:,document_indices[index]] += position_bias_score[index]
            visited.add(document_indices[index])
        
        return res # [1, vocab size]


    def decode(self, string_document, tokenized_document, tokenized_document_with_ref):
        document_embedding = torch.from_numpy(
            self.sent2vec_model.embed_sentence(string_document, pooling = "mean", drop_stopwords = True)).to(DEVICE).float() #[1, emb_size]
        
        self.fill_emb_phrase_vocab(tokenized_document_with_ref)
        
        semantic_score = torch.matmul(
            document_embedding,
            torch.transpose(self.SENT2VEC_EMB_WORD_VOCAB, 0, 1)
        )# [1, vocab_size]
        
        position_bias_score = torch.from_numpy(
            self.get_position_bias(tokenized_document)).to(DEVICE) # [1, vocab_size]
        

        self.document_embedding = document_embedding
        self.word_doc_sim = semantic_score #* position_bias_score

        # position_bias_score = self.get_position_bias(tokenized_document) # [1, vocab_size]
        # assert semantic_score.shape == position_bias_score.shape
        return self.word_doc_sim #semantic_score * position_bias_score

    def decode_mask_word(self, document):
        return None


    def decode_with_prev_steps(self, 
                               step: int,
                               current_phrases: List[List[str]], 
                               string_document,
                               tokenized_document,
                               tokenized_document_with_ref):
        current_phrase_length = step
    
        document_embedding = self.document_embedding #torch.from_numpy(self.sent2vec_model.embed_sentence(string_document, pooling = "mean")).to(DEVICE).float() # [1, emb_size]
        #self.fill_emb_phrase_vocab(tokenized_document_with_ref)

        current_phrases_embeddings = torch.from_numpy(self.sent2vec_model.embed_tokenized_sentences(
            current_phrases, pooling = "mean")).to(DEVICE).float() # [current phrases size, emb_size]
        
        # word_doc_sim = torch.matmul(
        #     document_embedding,
        #     torch.transpose(self.SENT2VEC_EMB_WORD_VOCAB, 0, 1)
        # ) # [1, vocab_size]
        word_doc_sim = self.word_doc_sim # [1, vocab_size]
        word_doc_sim = word_doc_sim.unsqueeze(1).repeat(1, len(current_phrases), 1) # [1, num current phrases, vocab_size]
        
        current_phrases_doc_sim = torch.matmul(
            document_embedding,
            torch.transpose(current_phrases_embeddings, 0, 1)
        ) # [1, num current phrases]
        #current_phrases_doc_sim = current_phrases_doc_sim.unsqueeze(-1).repeat(1, 1, self.SENT2VEC_EMB_WORD_VOCAB.size(0)) # [1, num current phrases, vocab_size]
        
        semantic_score = (word_doc_sim)[0] # [num current phrases, vocab size]
        # semantic_score[:, :EOS_IDX] /= (current_phrase_length + 1)
        # semantic_score[:, EOS_IDX + 1:] /= (current_phrase_length + 1)
        # semantic_score[:, EOS_IDX] /= current_phrase_length
        semantic_score[:, EOS_IDX] = current_phrases_doc_sim.reshape(-1)
        
        
        #position bias score
#         words_position_bias_score = self.get_position_bias(tokenized_document) # [1, vocab_size]
#         words_position_bias_score = words_position_bias_score.unsqueeze(1).repeat(1, len(current_phrases), 1) # [1, num current phrases, vocab_size]

#         current_phrases_position_bias = self.get_phrases_document_position_bias(current_phrases, tokenized_document) # [1, num_current_phrases]
#         current_phrases_position_bias = current_phrases_position_bias.unsqueeze(2).repeat(1, 1, self.SENT2VEC_EMB_WORD_VOCAB.shape[0])
        
#         position_bias_score = (words_position_bias_score + current_phrases_position_bias)[0] / (len(current_phrases[0]) + 1)
        
        #assert semantic_score.shape == position_bias_score.shape
        return semantic_score #* position_bias_score


#-----------------------------------------------------------------------------

SENT2VEC_CONFIG = {
    "vocab_path" : "/scratch/lamdo/unsupervised_keyphrase_prediction_2022/data/sent2vec/vocab.json",
    "uni_embs_path": "/scratch/lamdo/unsupervised_keyphrase_prediction_2022/data/sent2vec/uni_embs.npy"
}

S2V_MODEL = Sent2VecModel(SENT2VEC_CONFIG)


#------------------------------------------------------------------------------

KNN_INDEX, PHRASE_LIST = load_index("/scratch/lamdo/unsupervised_keyphrase_prediction_2022/src/phrase_retrieval/index_kp20k_kptimes_stackexchange")

#--------------------------------------------------------------------------------


def phraseness_model_decode(model, ys, pos_tag_src, pos_tag_ys, memory, src_decoder_tokens):
    tgt_mask = (generate_square_subsequent_mask(ys.size(1))
                        .type(torch.bool)).to(DEVICE)
    phrase_lm_prob = model.decoder(
        tgt = ys, pos_tag_src = pos_tag_src, pos_tag_tgt = pos_tag_ys, memory = memory, tgt_mask = tgt_mask, memory_mask = None, 
        tgt_padding_mask = None, memory_key_padding_mask = None, 
        src_decoder_tokens = src_decoder_tokens).detach()
    
                            
    # this for loop is to ensure that the model does not generate words that has been
    # generated at previous time step
    for i in range(ys.size(0)):
        phrase_lm_prob[i, -1, ys[i]] = 0
    # print("phrase_lm_prob shape", phrase_lm_prob.shape)
    return phrase_lm_prob


POS_TAGGER=nltk.tag.perceptron.PerceptronTagger()
def pos_tag_func(tokens):
    return POS_TAGGER.tag(tokens)


def get_pos_tag_of_targets(tgt, document_tokenized):
    # tgt is of shape [beam size, tgt length]
    res = []
    for index in range(tgt.size(0)):
        string_tgt = MODELS["vocab_transform"]["PHRASE"].lookup_tokens(list(tgt[index]), document_tokenized)
        pos_tag_tgt = MODELS["vocab_transform"]["POS_TAG"]\
        .lookup_indices([item[1] if item[0] not in ["<bos>", "<eos>"] else item[0] for item in pos_tag_func(string_tgt)])
        assert len(pos_tag_tgt) == tgt.size(1)
        res.append(pos_tag_tgt)
    
    res = torch.tensor(res).to(DEVICE).long()
    assert res.shape == tgt.shape
    return res



def normalize_score(score):
    #_score = score - torch.min(score, dim = 1)[0].view(-1, 1)
    _score = torch.relu(score) + 1e-5
    assert torch.all(_score >= 0), torch.min(_score)
    return torch.nn.functional.normalize(_score, 1, dim = 1)


def keyness_model_decode(model, string_document, tokenized_document, tokenized_document_with_ref):
    # vocab_embeddings: [vocab_size, emb_size] vocab_size = real vocab size + num position markers
    # document_embedding: [batch_size, emb_size]
    _res = model.decode(string_document, tokenized_document, tokenized_document_with_ref).to(DEVICE)
    return _res
    

def keyness_model_decode_with_prev_steps(step, model, current_phrases, string_document, tokenized_document, tokenized_document_with_ref):
    _res = model.decode_with_prev_steps(step, current_phrases, string_document, tokenized_document, tokenized_document_with_ref).to(DEVICE)
    return _res

import string
def detokenize(tokens):
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

def check_if_contain_absent_words(text_tokens: set, tokenized_candidate):
    if any(ctok not in text_tokens for ctok in tokenized_candidate): return True
    return False

def produce_input(tokenized_text, pos_tagged_text, tokenized_absent_phrases_list, pos_tagged_absent_phrases_list):
    text = detokenize(tokenized_text)
    phrases_list = [detokenize(item) for item in tokenized_absent_phrases_list]
    
    set_text_tokens = set(tokenized_text)
    indices =  [index for index,item in enumerate(tokenized_absent_phrases_list)]# if check_if_contain_absent_words(text_tokens = set_text_tokens, tokenized_candidate = item)]
    
    references = [tokenized_absent_phrases_list[index] + ["<sep>"] for index in indices]
    references = [tok for ref_tokens in references for tok in ref_tokens][:-1]
    
    pos_tagged_references = [pos_tagged_absent_phrases_list[index] + ["<sep>"] for index in indices]
    pos_tagged_references = [pos_tag for ref_pos_tags in pos_tagged_references for pos_tag in ref_pos_tags][:-1]
    
    assert len(references) == len(pos_tagged_references)
    
    inp = tokenized_text + ["<bor>"] + references + ["<eor>"]
    pos_tagged_inp = pos_tagged_text + ["<bor>"] + pos_tagged_references + ["<eor>"]
    
    assert len(inp) == len(pos_tagged_inp)
    
    return inp, pos_tagged_inp

def get_references(text, num = 20, mmr_lambda = 1):
    search_results = knn_search(text, KNN_INDEX, S2V_MODEL, PHRASE_LIST, 50, True, thresh = 0.7)[:num]
    more_phrases = search_results#list(mmr(search_results, num, mmr_lambda))
    
    tokenized_absent_phrases = [nltk.word_tokenize(phrase) for phrase in more_phrases]
    pos_tagged_absent_phrases = [[item[1] for item in pos_tag_func(tokenized_phrase)] for tokenized_phrase in tokenized_absent_phrases]
    return tokenized_absent_phrases, pos_tagged_absent_phrases


def _initial_state(phraseness_model, 
                   keyness_model, 
                   ys,
                   pos_tag_src,
                   pos_tag_ys, 
                   memory, 
                   src_decoder_tokens, 
                   string_document: str,
                   document_tokenized: list,
                   document_tokenized_with_ref: list,
                   lamb = 0.2,
                   beam_size = 200):
    # ys: [1, 1]
    # memory: [1, document length, encoder hidden size]
    # src_decoder_tokens: [1, document length]
    phraseness_prob = phraseness_model_decode(phraseness_model, ys, pos_tag_src, pos_tag_ys, memory, src_decoder_tokens) # [1, output length, vocab_size]
    phraseness_prob = phraseness_prob.view(-1, phraseness_prob.size(-1)) # [1, vocab_size]
    
    keyness_score = keyness_model_decode(keyness_model, 
                                         string_document = string_document,
                                         tokenized_document = document_tokenized, 
                                         tokenized_document_with_ref = document_tokenized_with_ref) # [1, vocab_size]
    keyness_prob = normalize_score(keyness_score)

    key_phrase_score = lamb * get_log_prob(phraseness_prob) + get_log_prob(keyness_prob) # [1, vocab_size]
    
    next_words_scores, next_words_indices = torch.topk(key_phrase_score, k = beam_size)

    new_ys = ys.repeat(beam_size, 1) # [beam_size, 1]
    new_ys = torch.cat([new_ys, next_words_indices.view(-1, 1)], dim = 1) # [beam_size, 2]
    pos_tag_new_ys = get_pos_tag_of_targets(new_ys, document_tokenized_with_ref)

    current_phrases = [MODELS["vocab_transform"]["PHRASE"].lookup_tokens(new_ys[i], document_tokenized_with_ref)[1:] for i in range(new_ys.shape[0])]
    # for i in range(new_ys.shape[0]):
    #     phrase = vocab_transform["PHRASE"].lookup_tokens(new_ys[i], document_tokenized_with_ref)[1:]
    #     current_phrases.append(phrase)

    return new_ys, pos_tag_new_ys, next_words_scores, current_phrases


def beam_search_step(step,
                     phraseness_model,
                      keyness_model, 
                      ys,
                      pos_tag_src,
                      pos_tag_ys,
                      memory,
                      src_decoder_tokens,
                      current_phrases,
                      prev_scores, 
                       string_document: str,
                      document_tokenized: list,
                      document_tokenized_with_ref,
                      lamb = 0.2):
    # ys.size(0) is current beam size
    # memory: [1, document length, encoder hidden size]
    # src_decoder_tokens: [1, document length]
    assert len(current_phrases) == ys.size(0)
    document = detokenize(document_tokenized)

    phraseness_prob = phraseness_model_decode(phraseness_model,
                                              ys = ys,
                                              pos_tag_src = pos_tag_src.repeat(ys.size(0), 1),
                                              pos_tag_ys = pos_tag_ys, 
                                              memory = memory.repeat(ys.size(0), 1, 1),
                                              src_decoder_tokens = src_decoder_tokens.repeat(ys.size(0), 1)) # [ys.size(0), ys.size(1), vocab size]
    phraseness_prob = phraseness_prob[:,-1,:] # [ys.size(0), vocab_size]
    
    keyness_score = keyness_model_decode_with_prev_steps(
        step, keyness_model, current_phrases, string_document = string_document, 
        tokenized_document = document_tokenized, 
        tokenized_document_with_ref = document_tokenized_with_ref) # [ys.size(0), vocab_size]
    keyness_prob = normalize_score(keyness_score)
    
    key_phrase_score = lamb * get_log_prob(phraseness_prob) + get_log_prob(keyness_prob) + prev_scores.view(-1, 1) # [ys.size(0), vocab_size]

    next_words_scores, next_words_indices = torch.topk(torch.flatten(key_phrase_score), k = ys.size(0))
    next_words_indices = np.array(np.unravel_index(next_words_indices.cpu().numpy(), key_phrase_score.shape)).T

    new_ys = [torch.cat([ys[next_words_indices[i][0]].view(1, -1),
                                     torch.ones(1, 1).type_as(ys.data).fill_(next_words_indices[i][1])], dim=1) for i in range(len(next_words_indices))]
    
    new_ys = torch.cat(new_ys, dim = 0) # [ys.size(0), ys.size(1) + 1]
    assert new_ys.shape == (ys.size(0), ys.size(1) + 1)
    pos_tag_new_ys = get_pos_tag_of_targets(new_ys, document_tokenized_with_ref)

    current_phrases = [MODELS["vocab_transform"]["PHRASE"].lookup_tokens(new_ys[i], document_tokenized_with_ref)[1:] for i in range(new_ys.shape[0])]

    return new_ys, pos_tag_new_ys, next_words_scores, current_phrases


def beam_search_inference(phraseness_model, 
                          keyness_model, 
                          document, 
                          max_len, 
                          start_symbol, 
                          end_symbol,
                          num_references, 
                          lamb, 
                          beam_size):
    tokenized_absent_phrases_list, pos_tagged_absent_phrases_list = get_references(document, num_references)
    
    with torch.no_grad():
        all_phrases = {}
        
        document_tokenized = nltk.tokenize.word_tokenize(document)[:512]
        document_pos_tagged = [item[1] for item in pos_tag_func(document_tokenized)]
        
        document_tokenized_with_ref, document_pos_tagged_with_ref = produce_input(
            document_tokenized, 
            document_pos_tagged, 
            tokenized_absent_phrases_list, 
            pos_tagged_absent_phrases_list)

        src, pos_tag_src = text_transform["TEXT"](document_tokenized_with_ref, document_pos_tagged_with_ref)
        src_decoder_tokens, _ = text_transform["PHRASE"](document_tokenized_with_ref, document_pos_tagged_with_ref, document_tokenized_with_ref)
        src = src.view(1,-1).to(DEVICE)
        pos_tag_src = pos_tag_src.view(1,-1).to(DEVICE)
        src_decoder_tokens = src_decoder_tokens.view(1,-1).to(DEVICE)

        num_tokens = src.size(1)
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(DEVICE)
        
        memory = phraseness_model.encoder(
            src = src, src_mask = src_mask
        )
        
        ys, pos_tag_ys, prev_scores, current_phrases = _initial_state(
            phraseness_model = phraseness_model, 
            keyness_model = keyness_model, 
            ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE), 
            pos_tag_src = pos_tag_src,
            pos_tag_ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE),
            memory = memory, 
            src_decoder_tokens = src_decoder_tokens, 
            string_document = document,
            document_tokenized = document_tokenized,
            document_tokenized_with_ref = document_tokenized_with_ref,
            lamb = lamb,
            beam_size = beam_size
        )

        for step in range(1, max_len):
            #print(current_phrases[:10])
            _ys, _pos_tag_ys, _prev_scores, _current_phrases = beam_search_step(
                step = step,
                phraseness_model = phraseness_model,
                keyness_model = keyness_model, 
                ys = ys,
                pos_tag_src = pos_tag_src,
                pos_tag_ys = pos_tag_ys,
                memory = memory,
                src_decoder_tokens = src_decoder_tokens,
                current_phrases = current_phrases,
                prev_scores = prev_scores,
                string_document = document,
                document_tokenized = document_tokenized,
                document_tokenized_with_ref = document_tokenized_with_ref,
                lamb = lamb
            )
            

            included_indices = []
            for i in range(_ys.size(0)):
                prev_scores = prev_scores.flatten()
                if _ys[i][-1] == end_symbol:
                    phrase = " ".join(_current_phrases[i]).replace("<bos>", "").replace("<eos>", "").replace(" - ", " ").strip().rstrip()
                    all_phrases[phrase] = {"score": _prev_scores[i].item(), "length": len(_ys[i]) - 1}
                else:
                    included_indices.append(i)
            if len(included_indices) == 0: break
            ys = _ys[included_indices, :]
            pos_tag_ys = _pos_tag_ys[included_indices, :]
            prev_scores = _prev_scores[included_indices]
            current_phrases = [_current_phrases[i] for i in included_indices]

        return all_phrases


#--------------------------------------------------------------------------------------

def apply_length_penalty(text, outputs, alpha):
    res = {}
    phrases = list(outputs.keys())
    for phrase in phrases:
        score = -outputs[phrase]["score"]
        length = outputs[phrase]["length"]
        
        length_norm = length + alpha
        
        res[phrase] = (score) / length_norm
    return list(sorted(res.items(), key = lambda x: x[1]))

#----------------------------------------------------------------------------------------


with open("/scratch/lamdo/unsupervised_keyphrase_prediction_2022/data/unsupervised/stable/phrase_counter_kp20k_kptimes_stackexchange.json") as f:
    PHRASE_COUNTER = json.load(f)


#------------------------------------------------------------------------------------------

from cleantext import clean

clean_func = lambda x: clean(
    x,
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lower=True,                     # lowercase text
    no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=True,                # replace all email addresses with a special token
    no_phone_numbers=True,         # replace all phone numbers with a special token
    no_numbers=False,               # replace all numbers with a special token
    no_digits=False,                # replace all digits with a special token
    no_currency_symbols=False,      # replace all currency symbols with a special token
    no_punct=False,                 # remove punctuations
    replace_with_url="<url>",
    replace_with_email="<email>",
    replace_with_phone_number="<phone>",
    lang="en"                       # set to 'de' for German special handling
)

#---------------------------------------------------------------------------------------------

def generate_for_dataset(data):
    all_keyphrases_scores = []

    present_kp_preds = []
    absent_kp_preds = []
    for index in range(len(data[:])):
        text = clean_func(data[index]["text"])
        if text != "":
            keyphrases_scores = beam_search_inference(phraseness_model = MODELS["transformer"], 
                                                    keyness_model = MODELS["keyness_model"], 
                                                    document = text, 
                                                    max_len = 6, 
                                                    start_symbol = BOS_IDX, 
                                                    end_symbol = EOS_IDX, 
                                                    num_references = NUM_REFERENCES,
                                                    lamb =  LAMB, 
                                                    beam_size = 100)
        else: keyphrases_scores = {}
        all_keyphrases_scores.append(keyphrases_scores)
    return all_keyphrases_scores


def generate_and_save_for_dataset(dataset_path, model_name, use_saved_predictions = True):
    folder_name = os.path.dirname(dataset_path)
    base=os.path.basename(dataset_path)
    dataset_file_name = os.path.splitext(base)[0]
    
    with open(dataset_path) as f:
        data = json.load(f)
    
    prediction_path = os.path.join(folder_name, f"predictions_{dataset_file_name}_{model_name}.json")
    
    if not os.path.exists(prediction_path) or not use_saved_predictions:
        all_keyphrases_scores = generate_for_dataset(data)
        with open(prediction_path, "w") as f:
            json.dump(all_keyphrases_scores, f)

    else:        
        with open(prediction_path) as f:
            all_keyphrases_scores = json.load(f)
        
    return all_keyphrases_scores


def find_position(tokenized_text, tokenized_phrase):
    for i, text_tok in enumerate(tokenized_text):
        if text_tok == tokenized_phrase[0]:
            flag = True
            for j in range(1, len(tokenized_phrase)):
                if i + j >= len(tokenized_text) or tokenized_phrase[j] != tokenized_text[i + j]:
                    flag = False
                    break
            if flag: return i
    return len(tokenized_text)
                

def adjust_score(text, keyphrases_scores, beta = 1.2):
    tokens = nltk.word_tokenize(text)
    # tokens_position_score = Counter()
    res = []
    # for i, tok in enumerate(tokens):
    #     if tok not in tokens_position_score: tokens_position_score[tok] = math.log(i + 2)

    for keyphrase, score in keyphrases_scores:
        if keyphrase in text:
            keyphrase_tokenized = nltk.word_tokenize(keyphrase)
            # keyphrase_pos_score = np.mean([tokens_position_score[tok] for tok in keyphrase_tokenized])
            keyphrase_pos_score = 1 + 1 / math.log2(find_position(tokens, keyphrase_tokenized) + 2)
            _score = score / keyphrase_pos_score
            res.append([keyphrase, _score])
        else: res.append([keyphrase, score / beta] if keyphrase in PHRASE_COUNTER else [keyphrase, score])

    return list(sorted(res, key = lambda x: x[1]))


def evaluate_for_each_hyperparam(all_hyperparams, all_keyphrases_scores, data):
    results = {}
    for alpha in all_hyperparams:#np.arange(-1, 1.01, 0.2):
        present_kp_preds = []
        absent_kp_preds = []
        for index in tqdm(range(len(data))):
            text = data[index]["text"]
            keyphrases_scores = apply_length_penalty(text, all_keyphrases_scores[index], alpha = alpha)
            if RERANK_BEAMSEARCH: keyphrases_scores = adjust_score(text, keyphrases_scores)

            present_keyphrases_scores = [item for item in keyphrases_scores if item[0] in text]
            present_keyphrases = [item[0] for item in present_keyphrases_scores]

            absent_keyphrases_scores = [item for item in keyphrases_scores if item[0] not in text]
            #absent_keyphrases_scores = [[item[0], item[1]] if item[0] in PHRASE_COUNTER else [item[0], item[1] * 1.2] for item in absent_keyphrases_scores]
            absent_keyphrases_scores = list(sorted(absent_keyphrases_scores, key = lambda x: x[1]))
            absent_keyphrases = [item[0] for item in absent_keyphrases_scores]

            present_kp_preds.append(present_keyphrases + [f"padding-{i}" for i in range(10)])
            absent_kp_preds.append(absent_keyphrases + [f"padding-{i}" for i in range(10)])

        # present_kp_results = evaluate_present_keyphrases(present_kp_preds, data)
        # absent_kp_results = evaluate_absent_keyphrases(absent_kp_preds, data)

        results[alpha] = {"present": present_kp_preds, "absent": absent_kp_preds}
    return results


def init_model(model_run_index = 1):
    global MODELS
    if MODELS.get("model_run_index") != model_run_index:
        print(f"Initializing model UOKG #{model_run_index}. This will be done only once")
        CKPT_PATH = f"/scratch/lamdo/unsupervised_keyphrase_prediction_2022/data/unsupervised_checkpoints/final/{model_run_index}/model.pth"

        checkpoint = torch.load(CKPT_PATH, map_location = DEVICE)
        CONFIG = checkpoint["config"]

        vocab_transform = {}

        vocab_transform["TEXT"] = checkpoint["text_vocab"]
        vocab_transform["PHRASE"] = checkpoint["phrase_vocab"]
        vocab_transform["POS_TAG"] = checkpoint["pos_tag_vocab"]


        TEXT_VOCAB_SIZE = len(vocab_transform["TEXT"])
        PHRASE_VOCAB_SIZE = len(vocab_transform["PHRASE"])
        POS_TAG_VOCAB_SIZE = len(vocab_transform["POS_TAG"])
        WORD_EMB_SIZE = CONFIG["word_emb_size"]
        POS_TAG_EMB_SIZE = CONFIG["pos_tag_emb_size"]
        NHEAD = CONFIG["nhead"]
        FFN_HID_DIM = CONFIG["ffn_hid_dim"]
        BATCH_SIZE = CONFIG["batch_size"]
        NUM_ENCODER_LAYERS = CONFIG["num_encoder_layers"]
        NUM_DECODER_LAYERS = CONFIG["num_decoder_layers"]
        NUM_POSITION_MARKERS = CONFIG["num_position_markers"]

        transformer = TransformerPointerGenerator(
            num_encoder_layers = NUM_ENCODER_LAYERS,
            num_decoder_layers = NUM_DECODER_LAYERS,
            word_emb_size = WORD_EMB_SIZE,
            pos_tag_emb_size = POS_TAG_EMB_SIZE,
            nhead = NHEAD,
            src_word_vocab_size = TEXT_VOCAB_SIZE,
            tgt_word_vocab_size = PHRASE_VOCAB_SIZE,
            pos_tag_vocab_size = POS_TAG_VOCAB_SIZE,
            num_position_markers = NUM_POSITION_MARKERS,
            dim_feedforward = FFN_HID_DIM,
            padding_idx = PAD_IDX)
        transformer.load_state_dict(checkpoint["transformer"])
        transformer = transformer.to(DEVICE)
        transformer.eval()

        keyness_model = KeynessModel(sent2vec_model = S2V_MODEL,
                             vocab_transform = vocab_transform,
                             tokenizer = nltk.tokenize.word_tokenize,
                             num_position_markers = CONFIG["num_position_markers"])


        MODELS = {
            "transformer": transformer,
            "vocab_transform": vocab_transform,
            "CONFIG": CONFIG,
            "keyness_model": keyness_model,
            "model_run_index": model_run_index
        }

        print("Done load model")


        


def keyphrase_generation_batch(docs, top_k = 10, alpha = 0.0, model_run_index = 0):
    lower_docs = [doc.lower() for doc in docs]
    init_model(model_run_index)
    all_keyphrases_scores =  generate_for_dataset([{"text": doc} for doc in lower_docs])

    for index in range(len(lower_docs)):
        text = lower_docs[index]
        keyphrases_scores = apply_length_penalty(text, all_keyphrases_scores[index], alpha = alpha)
        if RERANK_BEAMSEARCH: keyphrases_scores = adjust_score(text, keyphrases_scores)

        present_keyphrases_scores = [item for item in keyphrases_scores if item[0] in text]
        # present_keyphrases = [item[0] for item in present_keyphrases_scores]

        absent_keyphrases_scores = [item for item in keyphrases_scores if item[0] not in text]
        #absent_keyphrases_scores = [[item[0], item[1]] if item[0] in PHRASE_COUNTER else [item[0], item[1] * 1.2] for item in absent_keyphrases_scores]
        absent_keyphrases_scores = list(sorted(absent_keyphrases_scores, key = lambda x: x[1]))
        # absent_keyphrases = [item[0] for item in absent_keyphrases_scores]


    return {
        "present": present_keyphrases_scores,
        "absent": absent_keyphrases_scores
    }