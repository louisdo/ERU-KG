import json, os, torch, nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from typing import Iterable, List
from tqdm import tqdm




DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS = {}


from collections import Counter


# Define special symbols and indices, <unk> symbol will be in the last positions
PAD_IDX, BOS_IDX, EOS_IDX, URL_IDX, EMAIL_IDX, PHONE_IDX, TGT_UNK_IDX = 0, 1, 2, 3, 4, 5, 6
# TGT_UNK are for the tokens that neither appear in the vocab nor the text

class Lang:
    def __init__(self, num_position_markers = 1):
        assert num_position_markers >= 1
        self.num_position_markers = num_position_markers


    def build_vocab(self, 
                    data_iter: Iterable,
                    vocab_size: int):
        token_counter = Counter()
        for tokens in tqdm(yield_tokens(data_iter)):
            token_counter.update(tokens)
        # Make sure the tokens are in order of their indices to properly insert them in vocab
        special_symbols = ['<pad>', '<bos>', '<eos>', '<url>', '<email>', '<phone>', "<tgt_unk>"]
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



"""Model part"""

from torch import Tensor
import torch
import torch.nn as nn
import numpy as np
import math

import json, random, os
from typing import Iterable, List
from tqdm import tqdm


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, num_position_markers: int, emb_size):
        super(TokenEmbedding, self).__init__()

        self.vocab_size = vocab_size
        self.vocab_size_without_position_markers = vocab_size - num_position_markers + 1

        self.embedding = nn.Embedding(self.vocab_size_without_position_markers, emb_size,
                                      padding_idx = PAD_IDX)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        inp = torch.where(tokens >= self.vocab_size_without_position_markers,
                          torch.ones_like(tokens) * (self.vocab_size_without_position_markers - 1),
                          tokens)
        return self.embedding(inp.long()) * math.sqrt(self.emb_size)



class Encoder(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model = emb_size, 
                                                   nhead = nhead, 
                                                   dim_feedforward = dim_feedforward, 
                                                   dropout = dropout,
                                                   batch_first = True)
        encoder_norm = nn.LayerNorm(emb_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, 1, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor = None):
        return self.encoder(self.positional_encoding(self.src_tok_emb(src)), 
                            mask = src_mask,
                            src_key_padding_mask = src_key_padding_mask)
        


class Decoder(nn.Module):
    def __init__(self,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 tgt_vocab_size: int,
                 num_position_markers: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Decoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model = emb_size, 
                                                    nhead = nhead, 
                                                    dim_feedforward = dim_feedforward, 
                                                    dropout = dropout,
                                                    batch_first = True)
        decoder_norm = nn.LayerNorm(emb_size)
        self.tgt_vocab_size = tgt_vocab_size
        self.num_position_markers = num_position_markers
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.generator = nn.Linear(emb_size, tgt_vocab_size - num_position_markers)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, num_position_markers, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

        p_gen_input_size = emb_size * 2
        self.project_p_gens = nn.Linear(p_gen_input_size, 1)
        self.softmax = torch.nn.Softmax(dim = 2)

        # this is used to transform memory to compute copy distribution
        self.ff_h = nn.Sequential(
            nn.Linear(in_features = emb_size, out_features = emb_size),
            nn.ReLU(),
            nn.Linear(in_features = emb_size, out_features = emb_size)
        )
        self.ff_s = nn.Sequential(
            nn.Linear(in_features = emb_size, out_features = emb_size),
            nn.ReLU(),
            nn.Linear(in_features = emb_size, out_features = emb_size)
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor,
               memory_mask: Tensor, tgt_key_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        inp = self.tgt_tok_emb(tgt)
        output = self.decoder(self.positional_encoding(inp), 
                                memory = memory,
                                tgt_mask = tgt_mask,
                                memory_mask = memory_mask,
                                tgt_key_padding_mask = tgt_key_padding_mask,
                                memory_key_padding_mask = memory_key_padding_mask)

        return output

    def _get_copy_scores(self, 
                         memory: Tensor, 
                         decoder_outputs: Tensor, 
                         src_decoder_tokens: Tensor):
        # attn_weights: [batch_size, src_length, emb_size]
        # decoder_outputs: [batch_size, tgt_length, emb_size]
        # src_decoder_tokens: [batch_size, src_length], this is src but tokenized using the decoder's vocabulary


        batch_size, tgt_length = decoder_outputs.size(0), decoder_outputs.size(1)
        src_length = memory.size(1)
        
        decoder_states = decoder_outputs
        encoder_states = memory
        _attn_weights = torch.softmax(
            torch.bmm(
                self.ff_s(decoder_states), self.ff_h(encoder_states).transpose(2,1)
            ), 
            dim = 2
        ) # [batch size, tgt_length, src_length]
        assert _attn_weights.shape == (batch_size, tgt_length, src_length)


        attn_distribution_size = [batch_size, tgt_length, self.tgt_vocab_size]
        index = src_decoder_tokens[:, None, :]
        index = index.expand(batch_size, tgt_length, src_length)

        attn_dists = _attn_weights.new_zeros(attn_distribution_size) # [batch_size, tgt_length, tgt_vocab_size]
        attn_dists.scatter_add_(2, index, _attn_weights.float())

        return attn_dists

    
    def forward(self, 
                tgt: Tensor, 
                memory: Tensor, 
                tgt_mask: Tensor,
                memory_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor, 
                src_decoder_tokens: Tensor):
        decoder_outputs = self.decode(
            tgt = tgt, memory = memory, tgt_mask = tgt_mask,
            memory_mask = memory_mask, tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = memory_key_padding_mask
        )
        gen_scores = self.softmax(self.generator(decoder_outputs)) # [batch_size, tgt_length, tgt_vocab_size]
        padding_size = (gen_scores.size(0), gen_scores.size(1), self.num_position_markers)
        padding = gen_scores.new_zeros(padding_size) # [batch_size, tgt_length, tgt_vocab_size]
        gen_scores = torch.cat((gen_scores, padding), 2)
        assert gen_scores.size(2) == self.tgt_vocab_size

        copy_scores = self._get_copy_scores(
            memory = memory, decoder_outputs = decoder_outputs, 
            src_decoder_tokens = src_decoder_tokens
        )

        p_gen_predictors = torch.cat([self.positional_encoding(self.tgt_tok_emb(tgt)), decoder_outputs], dim = 2) # [batch_size, tgt_length, emb_size]
        p_gens = torch.sigmoid(self.project_p_gens(p_gen_predictors)) # [batch_size, tgt_length, 1]

        assert copy_scores.shape == gen_scores.shape
        final_probs = torch.mul(gen_scores.float(), p_gens) + torch.mul(copy_scores.float(), (1 - p_gens))
        return final_probs




class TransformerPointerGenerator(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 num_position_markers: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(TransformerPointerGenerator, self).__init__()
        
        self.encoder = Encoder(num_encoder_layers = num_encoder_layers,
                                emb_size = emb_size,
                                nhead = nhead,
                                src_vocab_size = src_vocab_size,
                                dim_feedforward = dim_feedforward,
                                dropout = dropout)
        
        self.decoder = Decoder(num_decoder_layers = num_decoder_layers,
                                emb_size = emb_size,
                                nhead = nhead,
                                tgt_vocab_size = tgt_vocab_size,
                                num_position_markers = num_position_markers,
                                dim_feedforward = dim_feedforward,
                                dropout = dropout)

    def forward(self,
                src: Tensor,
                src_decoder_tokens: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
    
        memory = self.encoder(src = src, src_mask = src_mask, src_key_padding_mask = src_padding_mask)
        output = self.decoder(tgt = tgt, memory = memory, tgt_mask = tgt_mask, memory_mask = None, 
                              tgt_padding_mask = tgt_padding_mask, memory_key_padding_mask = memory_key_padding_mask, src_decoder_tokens = src_decoder_tokens)
        return output


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def get_log_prob(prob):
    return prob.clamp(1e-10, 1.0).log()



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

def transform_text(txt_input):
    res = MODELS["TOKENIZER"](txt_input)[:MODELS["CONFIG"]["num_position_markers"]]
    res = MODELS["vocab_transform"]["TEXT"].lookup_indices(res)
    res = tensor_transform(res)
    return res

def transform_phrase(phrase_input, txt_input = None):
    res = MODELS["TOKENIZER"](phrase_input)[:MODELS["CONFIG"]["num_position_markers"]]
    if txt_input:
        res = MODELS["vocab_transform"]["PHRASE"].lookup_indices(res, 
                                                       MODELS["TOKENIZER"](txt_input)[:MODELS["CONFIG"]["num_position_markers"]])
    else:
        res = MODELS["vocab_transform"]["PHRASE"].lookup_indices(res)
    res = tensor_transform(res)
    return res

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
text_transform["TEXT"] = transform_text
text_transform["PHRASE"] = transform_phrase

        
NUM_DOC_KP20K = 514154
LAMBDA = 1.4

from nltk.stem.porter import PorterStemmer
STEMMER = PorterStemmer()
CACHE_STEMMING = {}

def _stem_word(word):
    if word not in CACHE_STEMMING:
        stemmed_word = STEMMER.stem(word)
        CACHE_STEMMING[word] = stemmed_word
    else: stemmed_word = CACHE_STEMMING[word]
    return stemmed_word

def _stem_phrase(phrase_tokenized: list):
    return [_stem_word(tok) for tok in phrase_tokenized]

_stem_text = _stem_phrase


import math
from sklearn.preprocessing import normalize


def semantic_score(text, candidates):
    if len(candidates) == 0: return []
    text_embedding = normalize(sent2vec_model.embed_sentences([text]), axis = 1)
    candidates_embeddings = normalize(sent2vec_model.embed_sentences(candidates), axis = 1)

    scores = (np.dot(text_embedding, candidates_embeddings.transpose(1, 0))[0] + 1) / 2

    candidates_scores = {candidates[i]: scores[i] for i in range(len(candidates))}#[[candidates[i], scores[i]] for i in range(len(candidates))]
    return candidates_scores
        
        

def lexical_score(text, candidates):
    text_token = nltk.word_tokenize(text)
    text_stemmed_token = _stem_text(text_token)
    text_stemmed_token_counter = Counter(text_stemmed_token)
    
    candidates_scores = {c:lexical_score_each_candidate(text_stemmed_token_counter, c) for c in candidates}
    return candidates_scores


def rank(text, candidates):
    candidates_semantic_scores = semantic_score(text, candidates)
    candidates_lexical_scores = lexical_score(text, candidates)
    
    candidates_scores = {c:(candidates_semantic_scores[c]**LAMBDA * candidates_lexical_scores[c]) ** 0.5 for c in candidates}
        
    res = [[k,v] for k,v in candidates_scores.items()]
    return sorted(res, key = lambda x: -x[1])


from sklearn.metrics.pairwise import cosine_similarity as cosine

def phraseness_model_decode(model, ys, memory, src_decoder_tokens):
    tgt_mask = (generate_square_subsequent_mask(ys.size(1))
                        .type(torch.bool)).to(DEVICE)
    phrase_lm_prob = model.decoder(ys, memory, tgt_mask,
                            memory_mask = None,
                            tgt_padding_mask = None,
                            memory_key_padding_mask = None, 
                            src_decoder_tokens = src_decoder_tokens).detach()
                            
    # this for loop is to ensure that the model does not generate words that has been
    # generated at previous time step
    for i in range(ys.size(0)):
        phrase_lm_prob[i, -1, ys[i]] = 0
    # print("phrase_lm_prob shape", phrase_lm_prob.shape)
    return phrase_lm_prob



def _initial_state(phraseness_model, 
                   ys, 
                   memory, 
                   src_decoder_tokens, 
                   document: str,
                   beam_size = 200):
    # ys: [1, 1]
    # memory: [1, document length, encoder hidden size]
    # src_decoder_tokens: [1, document length]
    document_tokenized = MODELS["TOKENIZER"](document)

    phraseness_prob = phraseness_model_decode(phraseness_model, ys, memory, src_decoder_tokens) # [1, output length, vocab_size]
    phraseness_prob = phraseness_prob.view(-1, phraseness_prob.size(-1)) # [1, vocab_size]

    key_phrase_score = get_log_prob(phraseness_prob)

    next_words_scores, next_words_indices = torch.topk(key_phrase_score, k = beam_size, dim=1)

    new_ys = ys.repeat(beam_size, 1) # [beam_size, 1]
    new_ys = torch.cat([new_ys, next_words_indices.view(-1, 1)], dim = 1) # [beam_size, 2]

    current_phrases = []
    for i in range(new_ys.shape[0]):
        phrase = MODELS["vocab_transform"]["PHRASE"].lookup_tokens(new_ys[i], document_tokenized)[1:]
        current_phrases.append(phrase)

    return new_ys, next_words_scores, current_phrases


def beam_search_step(step,
                     phraseness_model, 
                      ys,
                      memory,
                      src_decoder_tokens,
                      current_phrases,
                      prev_scores, 
                      document: str):
    # ys.size(0) is current beam size
    # memory: [1, document length, encoder hidden size]
    # src_decoder_tokens: [1, document length]
    assert len(current_phrases) == ys.size(0)
    document_tokenized = MODELS["TOKENIZER"](document)

    phraseness_prob = phraseness_model_decode(phraseness_model,
                                              ys = ys, 
                                              memory = memory.repeat(ys.size(0), 1, 1),
                                              src_decoder_tokens = src_decoder_tokens.repeat(ys.size(0), 1)) # [ys.size(0), ys.size(1), vocab size]
    phraseness_prob = phraseness_prob[:,-1,:] # [ys.size(0), vocab_size]
    #_, top_fluent_words_indices = torch.topk(phraseness_prob,dim = 1, k = 500)

    key_phrase_score = get_log_prob(phraseness_prob) + prev_scores.view(-1, 1)
    
    next_words_scores, next_words_indices = torch.topk(torch.flatten(key_phrase_score), k = ys.size(0))
    next_words_indices = np.array(np.unravel_index(next_words_indices.cpu().numpy(), key_phrase_score.shape)).T

    new_ys = []
    for i in range(len(next_words_indices)):
        to_append = ys[next_words_indices[i][0]].view(1, -1)
        to_append = torch.cat([to_append,
                    torch.ones(1, 1).type_as(ys.data).fill_(next_words_indices[i][1])], dim=1)
        new_ys.append(to_append)
    new_ys = torch.cat(new_ys, dim = 0) # [ys.size(0), ys.size(1) + 1]
    assert new_ys.shape == (ys.size(0), ys.size(1) + 1)

    current_phrases = []
    for i in range(new_ys.shape[0]):
        phrase = MODELS["vocab_transform"]["PHRASE"].lookup_tokens(new_ys[i], document_tokenized)[1:]
        current_phrases.append(phrase)

    return new_ys, next_words_scores, current_phrases


def beam_search_inference(phraseness_model, 
                          document, 
                          max_len, 
                          start_symbol, 
                          end_symbol, 
                          beam_size):
    with torch.no_grad():
        all_phrases = {}
        document_tokenized = MODELS["TOKENIZER"](document)[:MODELS["CONFIG"]["num_position_markers"]]
        document = " ".join(document_tokenized)

        src = text_transform["TEXT"](document).view(1,-1).to(DEVICE)
        src_decoder_tokens = text_transform["PHRASE"](document, document).view(1,-1).to(DEVICE)
        num_tokens = src.size(1)
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(DEVICE)
        memory = phraseness_model.encoder(src, src_mask)

        ys, prev_scores, current_phrases = _initial_state(phraseness_model = phraseness_model, 
                                                          ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE), 
                                                          memory = memory, 
                                                          src_decoder_tokens = src_decoder_tokens, 
                                                          document = document,
                                                          beam_size = beam_size)
        for step in range(1, max_len):
            _ys, _prev_scores, _current_phrases = beam_search_step(step = step,
                                                                    phraseness_model = phraseness_model, 
                                                                    ys = ys,
                                                                    memory = memory,
                                                                    src_decoder_tokens = src_decoder_tokens,
                                                                    current_phrases = current_phrases,
                                                                    prev_scores = prev_scores, 
                                                                    document = document)

            included_indices = []
            for i in range(_ys.size(0)):
                if _ys[i][-1] == end_symbol:
                    phrase = " ".join(_current_phrases[i]).replace("<bos>", "").replace("<eos>", "").replace(" - ", " ").strip().rstrip()
                    length_norm = (len(_ys[i]) - 1)
                    #print(phrase, length_reward, length_norm)
                    all_phrases[phrase] = {"score": _prev_scores[i].item(), "length": len(_ys[i]) - 1}
                else:
                    included_indices.append(i)
            if len(included_indices) == 0: break
            ys = _ys[included_indices, :]
            prev_scores = _prev_scores[included_indices]
            current_phrases = [_current_phrases[i] for i in included_indices]

        return all_phrases#list(sorted(all_phrases.items(), key = lambda x: -x[1]))


def apply_length_penalty(text, outputs, alpha):
    res = {}
    phrases = list(outputs.keys())
    for phrase in phrases:
        score = -outputs[phrase]["score"]
        length = outputs[phrase]["length"]
        
        length_norm = length + alpha
        
        res[phrase] = (score) / length_norm
    return list(sorted(res.items(), key = lambda x: x[1]))


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






def generate_for_dataset(data):
    all_keyphrases_scores = []

    present_kp_preds = []
    absent_kp_preds = []
    for index in tqdm(range(len(data[:]))):
        text = clean_func(data[index]["text"])
        if text != "":
            keyphrases_scores = beam_search_inference(phraseness_model = MODELS["transformer"], 
                                document = text, 
                                max_len = 6, 
                                start_symbol = BOS_IDX, 
                                end_symbol = EOS_IDX, 
                                beam_size = 100)
        else: keyphrases_scores = {}
        all_keyphrases_scores.append(keyphrases_scores)
    return all_keyphrases_scores



def init_model(model_run_index):
    global MODELS
    if MODELS.get("model_run_index") != model_run_index:
        CKPT_PATH = f"/scratch/lamdo/unsupervised_keyphrase_prediction_2022/data/supervised_checkpoints/final/{model_run_index}/supervised.pth"

        assert os.path.exists(CKPT_PATH)
        CKPT = torch.load(CKPT_PATH, map_location = DEVICE)


        CONFIG = CKPT["config"]


        TOKENIZER = lambda x: nltk.word_tokenize(x)


        CKPT_FOLDER = os.path.dirname(CKPT_PATH)

        vocab_transform = {}

        vocab_transform["TEXT"] = CKPT["text_vocab"]
        vocab_transform["PHRASE"] = CKPT["phrase_vocab"]


        TEXT_VOCAB_SIZE = len(vocab_transform["TEXT"])
        PHRASE_VOCAB_SIZE = len(vocab_transform["PHRASE"])
        EMB_SIZE = CONFIG["emb_size"]
        NHEAD = CONFIG["nhead"]
        FFN_HID_DIM = CONFIG["ffn_hid_dim"]
        NUM_ENCODER_LAYERS = CONFIG["num_encoder_layers"]
        NUM_DECODER_LAYERS = CONFIG["num_decoder_layers"]
        NUM_POSITION_MARKERS = CONFIG["num_position_markers"]

        transformer = TransformerPointerGenerator(num_encoder_layers = NUM_ENCODER_LAYERS,
                                                num_decoder_layers = NUM_DECODER_LAYERS,
                                                emb_size = EMB_SIZE,
                                                nhead = NHEAD,
                                                src_vocab_size = TEXT_VOCAB_SIZE,
                                                tgt_vocab_size = PHRASE_VOCAB_SIZE,
                                                num_position_markers = NUM_POSITION_MARKERS,
                                                dim_feedforward = FFN_HID_DIM)
        transformer.load_state_dict(CKPT["transformer"])
        transformer = transformer.to(DEVICE)
        transformer.eval()


        MODELS = {
                    "transformer": transformer,
                    "vocab_transform": vocab_transform,
                    "CONFIG": CONFIG,
                    "model_run_index": model_run_index,
                    "TOKENIZER": TOKENIZER
                }
        
        print("Done load model")


def keyphrase_generation_batch(docs, top_k = 10, alpha = 0.0, model_run_index = 1):
    init_model(model_run_index)

    lower_docs = [str(doc).lower() for doc in docs]

    all_keyphrases_scores =  generate_for_dataset([{"text": doc} for doc in lower_docs])

    for index in range(len(lower_docs)):
        text = lower_docs[index]
        keyphrases_scores = apply_length_penalty(text, all_keyphrases_scores[index], alpha = alpha)

        present_keyphrases_scores = [item for item in keyphrases_scores if item[0] in text]
        # position_biases = position_score(text, [item[0] for item in present_keyphrases_scores])
        # #print(position_biases)
        # present_keyphrases_scores = list(sorted([[item[0], item[1] * position_biases[item[0]]] for item in present_keyphrases_scores], key = lambda x: -x[1]))
        # present_keyphrases = [item[0] for item in present_keyphrases_scores]

        absent_keyphrases_scores = [item for item in keyphrases_scores if item[0] not in text]
        # absent_keyphrases_scores = [[item[0], item[1] / 2] if item[0] in PHRASE_COUNTER else [item[0], item[1]] for item in absent_keyphrases_scores]
        absent_keyphrases_scores = list(sorted(absent_keyphrases_scores, key = lambda x: x[1]))
        # absent_keyphrases = [item[0] for item in absent_keyphrases_scores]

        # present_kp_preds.append(present_keyphrases + [f"padding-{i}" for i in range(10)])
        # absent_kp_preds.append(absent_keyphrases + [f"padding-{i}" for i in range(10)])


    return {
        "present": present_keyphrases_scores,
        "absent": absent_keyphrases_scores
    }