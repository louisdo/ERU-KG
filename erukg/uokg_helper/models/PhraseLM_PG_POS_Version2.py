from torch import Tensor
import torch, copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Transformer
import math
from sklearn.metrics.pairwise import cosine_similarity as cosine
from typing import Optional, Any, Union, Callable

import json, random, os
# from torchtext.vocab import build_vocab_from_iterator
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
    def __init__(self, vocab_size: int, num_position_markers: int, emb_size: int, padding_idx: int):
        super(TokenEmbedding, self).__init__()

        self.vocab_size = vocab_size
        self.vocab_size_without_position_markers = vocab_size - num_position_markers + 1

        self.embedding = nn.Embedding(self.vocab_size_without_position_markers, emb_size,
                                      padding_idx = padding_idx)
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
                 dropout: float = 0.1,
                 padding_idx: int = 0):
        super(Encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model = emb_size, 
                                                   nhead = nhead, 
                                                   dim_feedforward = dim_feedforward, 
                                                   dropout = dropout,
                                                   batch_first = True)
        encoder_norm = nn.LayerNorm(emb_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, 1, emb_size, padding_idx = padding_idx)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor = None):
        return self.encoder(self.positional_encoding(self.src_tok_emb(src)), 
                            mask = src_mask,
                            src_key_padding_mask = src_key_padding_mask)
        


class Decoder(nn.Module):
    def __init__(self,
                 num_decoder_layers: int,
                 word_emb_size: int,
                 pos_tag_emb_size: int,
                 nhead: int,
                 tgt_word_vocab_size: int,
                 pos_tag_vocab_size,
                 num_position_markers: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 padding_idx: int = 0):
        super(Decoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model = word_emb_size, 
                                                    nhead = nhead, 
                                                    dim_feedforward = dim_feedforward, 
                                                    dropout = dropout,
                                                    batch_first = True)
        decoder_norm = nn.LayerNorm(word_emb_size)
        self.tgt_word_vocab_size = tgt_word_vocab_size
        self.num_position_markers = num_position_markers
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.generator = nn.Linear(word_emb_size, tgt_word_vocab_size - num_position_markers)
        self.tgt_tok_emb = TokenEmbedding(tgt_word_vocab_size, num_position_markers, word_emb_size, padding_idx = padding_idx)
        self.pos_tag_emb = nn.Embedding(pos_tag_vocab_size, pos_tag_emb_size, padding_idx = padding_idx)
        
        self.positional_encoding = PositionalEncoding(
            word_emb_size, dropout=dropout)

        p_gen_input_size = (word_emb_size) * 2
        self.project_p_gens = nn.Linear(p_gen_input_size, 1)
        self.softmax = torch.nn.Softmax(dim = 2)

        # this is used to transform memory to compute copy distribution
        #self.memory_transform = nn.Linear(in_features = word_emb_size, out_features = word_emb_size)
        self.ff_h = nn.Sequential(
            nn.Linear(in_features = word_emb_size + pos_tag_emb_size, out_features = word_emb_size + pos_tag_emb_size),
            nn.ReLU(),
            nn.Linear(in_features = word_emb_size + pos_tag_emb_size, out_features = word_emb_size + pos_tag_emb_size)
        )
        self.ff_s = nn.Sequential(
            nn.Linear(in_features = word_emb_size + pos_tag_emb_size, out_features = word_emb_size + pos_tag_emb_size),
            nn.ReLU(),
            nn.Linear(in_features = word_emb_size + pos_tag_emb_size, out_features = word_emb_size + pos_tag_emb_size)
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
                         pos_tag_src: Tensor,
                         pos_tag_tgt: Tensor,
                         memory: Tensor, 
                         decoder_outputs: Tensor, 
                         src_decoder_tokens: Tensor):
        # memory: [batch_size, src_length, word_emb_size]
        # pos_tag_src_emb: [batch_size, src_length, pos_tag_emb_size]
        # decoder_outputs: [batch_size, tgt_length, word_emb_size]
        # pos_tag_tgt_emb: [batch_size, tgt_length, pos_tag_emb_size]
        # src_decoder_tokens: [batch_size, src_length], this is src but tokenized using the decoder's vocabulary


        batch_size, tgt_length = decoder_outputs.size(0), decoder_outputs.size(1)
        src_length = memory.size(1)
        
        pos_tag_tgt_emb = self.pos_tag_emb(pos_tag_tgt)
        pos_tag_src_emb = self.pos_tag_emb(pos_tag_src)
        
        decoder_states = torch.cat([decoder_outputs, pos_tag_tgt_emb], dim = 2) # [batch_size, tgt_length, word_emb_size + pos_tag_emb_size]
        encoder_states = torch.cat([memory, pos_tag_src_emb], dim = 2) # [batch_size, src_length, word_emb_size + pos_tag_emb_size]
        _attn_weights = torch.softmax(
            torch.bmm(
                self.ff_s(decoder_states), self.ff_h(encoder_states).transpose(2,1)
            ), 
            dim = 2
        ) # [batch size, tgt_length, src_length]
        assert _attn_weights.shape == (batch_size, tgt_length, src_length)


        attn_distribution_size = [batch_size, tgt_length, self.tgt_word_vocab_size]
        index = src_decoder_tokens[:, None, :]
        index = index.expand(batch_size, tgt_length, src_length)

        attn_dists = _attn_weights.new_zeros(attn_distribution_size) # [batch_size, tgt_length, tgt_vocab_size]
        attn_dists.scatter_add_(2, index, _attn_weights.float())

        return attn_dists

    
    def forward(self, 
                tgt: Tensor, 
                pos_tag_src,
                pos_tag_tgt,
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
        assert gen_scores.size(2) == self.tgt_word_vocab_size

        copy_scores = self._get_copy_scores(
            pos_tag_src = pos_tag_src, pos_tag_tgt = pos_tag_tgt,
            memory = memory, decoder_outputs = decoder_outputs, 
            src_decoder_tokens = src_decoder_tokens
        )

        p_gen_predictors = torch.cat([self.positional_encoding(self.tgt_tok_emb(tgt)), decoder_outputs], dim = 2) # [batch_size, tgt_length, word_emb_size * 2]
        p_gens = torch.sigmoid(self.project_p_gens(p_gen_predictors)) # [batch_size, tgt_length, 1]

        assert copy_scores.shape == gen_scores.shape
        final_probs = torch.mul(gen_scores.float(), p_gens) + torch.mul(copy_scores.float(), (1 - p_gens))
        return final_probs #[batch_size, tgt_length]




class TransformerPointerGenerator(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 word_emb_size: int,
                 pos_tag_emb_size: int,
                 nhead: int,
                 src_word_vocab_size: int,
                 tgt_word_vocab_size: int,
                 pos_tag_vocab_size:int,
                 num_position_markers: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 padding_idx: int = 0):
        super(TransformerPointerGenerator, self).__init__()
        
        self.encoder = Encoder(
            num_encoder_layers = num_encoder_layers,
             emb_size = word_emb_size,
             nhead = nhead,
             src_vocab_size = src_word_vocab_size,
             dim_feedforward = dim_feedforward,
             dropout = dropout
        )
        
        self.decoder = Decoder(
            num_decoder_layers = num_decoder_layers,
             word_emb_size = word_emb_size,
             pos_tag_emb_size = pos_tag_emb_size,
             nhead = nhead,
             tgt_word_vocab_size = tgt_word_vocab_size,
             pos_tag_vocab_size = pos_tag_vocab_size,
             num_position_markers = num_position_markers,
             dim_feedforward = dim_feedforward,
             dropout = dropout,
             padding_idx = padding_idx
        )

    def forward(self,
                src: Tensor,
                pos_tag_src: Tensor,
                src_decoder_tokens: Tensor,
                tgt: Tensor,
                pos_tag_tgt,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
    
        memory = self.encoder(src = src, src_mask = src_mask, src_key_padding_mask = src_padding_mask)
        output = self.decoder(tgt = tgt, pos_tag_src = pos_tag_src, pos_tag_tgt = pos_tag_tgt, memory = memory, tgt_mask = tgt_mask, memory_mask = None, 
                              tgt_padding_mask = tgt_padding_mask, memory_key_padding_mask = memory_key_padding_mask, src_decoder_tokens = src_decoder_tokens)
        return output