import json, os
from datasets import load_dataset
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from typing import *

from src.two_stage_keyphrase_extraction_with_splade import keyphrase_extraction as splade_based_keyphrase_extraction
from src.embedrank_keyphrase_extraction import embedrank_keyphrase_extraction, embed_sentences_sentence_transformer, embed_sentences_sent2vec
from src.multipartiterank import keyphrase_extraction as multipartiterank_keyphrase_extraction
# from src.keyBART import generate_keywords as keybart_keyphrase_generation
from src.process_dataset import process_dataset
# from src.ukg import generate_keyphrases as ukg_keyphrase_generation
from src.retrieval_based_ukg import keyphrase_generation as retrieval_based_ukg_keyphrase_generation
from src.textrank import keyphrase_extraction as textrank_keyphrase_extraction
from src.nounphrase_extractor import nounphrase_extraction_as_keyphrase_generation
# from pyserini.search.lucene import LuceneSearcher

# this is for uokg
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
from src.uokg import keyphrase_generation_batch as keyphrase_generation_batch_uokg

RETRIEVAL_DATASETS = ["nq320k", "scirepeval_search", "scirepeval_search_validation_evaluation"]

RESULTS_FOLDER = os.environ["RESULTS_FOLDER"]

MODEL_TO_USE = os.environ["MODEL_TO_USE"]
DATASET_TO_USE = os.environ["DATASET_TO_USE"]
TOP_KS_TO_EVAL= [3,5,10]

RESULT_FILE = os.path.join(RESULTS_FOLDER, f"{DATASET_TO_USE}--{MODEL_TO_USE}.json")


def do_keyphrase_extraction(doc, top_k = 10):
    if MODEL_TO_USE == "embedrank_sent2vec":
        return embedrank_keyphrase_extraction(doc, embed_func=embed_sentences_sent2vec, top_k = top_k)
    
    elif MODEL_TO_USE == "embedrank_sentence_transformers_all-MiniLM-L6-v2":
        return embedrank_keyphrase_extraction(doc, embed_func=embed_sentences_sentence_transformer, top_k = top_k)

    elif MODEL_TO_USE == "embedrank_sentence_transformers_multi-qa-MiniLM-L6-cos-v1":
        return embedrank_keyphrase_extraction(doc, embed_func=lambda x: embed_sentences_sentence_transformer(x, model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"))
    elif MODEL_TO_USE == "embedrank_sentence_transformers_all-MiniLM-L12-v2":
        return embedrank_keyphrase_extraction(doc, embed_func=lambda x: embed_sentences_sentence_transformer(x, model_name = "sentence-transformers/all-MiniLM-L12-v2"))

    elif MODEL_TO_USE == "multipartiterank":
        return multipartiterank_keyphrase_extraction(doc, top_k)
    
    elif MODEL_TO_USE == "textrank":
        return textrank_keyphrase_extraction(doc, top_k)

    # elif MODEL_TO_USE == "keybart":
    #     return keybart_keyphrase_generation(doc, top_k = top_k)
    
    elif MODEL_TO_USE == "splade_based_splade-cocondenser-selfdistil":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name = "splade-cocondenser-selfdistil")
    
    elif MODEL_TO_USE == "splade_based_splade-cocondenser-ensembledistil":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="splade-cocondenser-ensembledistil")
    elif MODEL_TO_USE == "splade_based_splade-cocondenser-ensembledistil_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="splade-cocondenser-ensembledistil", 
                                                 apply_position_penalty=True, length_penalty = -0.25)
    
    elif MODEL_TO_USE == "splade_based_custom_trained_pubmedqa+specter":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_pubmedqa+specter")
    elif MODEL_TO_USE == "splade_based_custom_trained_msmarco":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_msmarco")
    elif MODEL_TO_USE == "splade_based_custom_trained_scirepeval_search":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_scirepeval_search")
    elif MODEL_TO_USE == "splade_based_custom_trained_scirepeval_search_position_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_scirepeval_search",
                                                 apply_position_penalty=True)
    elif MODEL_TO_USE == "splade_based_custom_trained_scirepeval_search_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_scirepeval_search",
                                                 apply_position_penalty=True, length_penalty=-0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_scirepeval_highly_influential_citation":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_scirepeval_highly_influential_citation")
    elif MODEL_TO_USE == "splade_based_custom_trained_scirepeval_highly_influential_citation->search":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_scirepeval_highly_influential_citation->search")
    elif MODEL_TO_USE == "splade_based_custom_trained_scirepeval_highly_influential_citation+search":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_scirepeval_highly_influential_citation+search")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_citation_context":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_citation_context")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_citation_context->scirepeval_search":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_citation_context->scirepeval_search")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_citation_context+scirepeval_search":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_citation_context+scirepeval_search")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxiv_citation_context_random_negsampling_only":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxiv_citation_context_random_negsampling_only")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxiv_intro_related_work":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxiv_intro_related_work")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_full_paper":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_full_paper")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_full_paper->scirepeval_search":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_full_paper->scirepeval_search")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_full_paper->scirepeval_search_position_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_full_paper->scirepeval_search", 
                                                 apply_position_penalty=True)
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_full_paper->scirepeval_search_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_full_paper->scirepeval_search", 
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_full_paper_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_full_paper",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_full_paper_v2":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_full_paper_v2")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_full_paper_v2_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_full_paper_v2",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_scirepeval_search_v2":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_scirepeval_search_v2")
    elif MODEL_TO_USE == "splade_based_custom_trained_scirepeval_search_v2_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_scirepeval_search_v2",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_full_paper_v2->scirepeval_search_v2":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_full_paper_v2->scirepeval_search_v2")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_full_paper_v2->scirepeval_search_v2_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_full_paper_v2->scirepeval_search_v2",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_intro_relatedwork_1citationpersentence":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_intro_relatedwork_1citationpersentence")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_intro_relatedwork_1citationpersentence_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_intro_relatedwork_1citationpersentence",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_intro_relatedwork_1citationpersentence->scirepeval_search":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_intro_relatedwork_1citationpersentence->scirepeval_search")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_intro_relatedwork_1citationpersentence->scirepeval_search_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_intro_relatedwork_1citationpersentence->scirepeval_search",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_full_paper_1citationpersentence":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_full_paper_1citationpersentence")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_full_paper_1citationpersentence_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_full_paper_1citationpersentence",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_splade_doc_unarxive_full_paper_1citationpersentence":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_splade_doc_unarxive_full_paper_1citationpersentence")
    elif MODEL_TO_USE == "splade_based_custom_trained_splade_doc_unarxive_full_paper_1citationpersentence_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_splade_doc_unarxive_full_paper_1citationpersentence",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_splade_doc_unarxive_full_paper_1citationpersentence->scirepeval_search":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_splade_doc_unarxive_full_paper_1citationpersentence->scirepeval_search")
    elif MODEL_TO_USE == "splade_based_custom_trained_splade_doc_unarxive_full_paper_1citationpersentence->scirepeval_search_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_splade_doc_unarxive_full_paper_1citationpersentence->scirepeval_search",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_full_paper_1citationpersentence->scirepeval_search":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_full_paper_1citationpersentence->scirepeval_search")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_full_paper_1citationpersentence->scirepeval_search_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_full_paper_1citationpersentence->scirepeval_search",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_scirepeval_search->unarxive_intro_relatedwork_1citationpersentence":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_scirepeval_search->unarxive_intro_relatedwork_1citationpersentence")
    elif MODEL_TO_USE == "splade_based_custom_trained_scirepeval_search->unarxive_intro_relatedwork_1citationpersentence_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_scirepeval_search->unarxive_intro_relatedwork_1citationpersentence",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search_v2":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search_v2")
    elif MODEL_TO_USE == "splade_based_custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search_v2_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search_v2",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references")
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v2":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v2")
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v2_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v2",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v3":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v3")
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v3_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v3",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v4":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v4")
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v4_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v4",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v5":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v5")
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v5_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v5",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v6":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v6")
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v6_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v6",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v7":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v7")
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v7_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v7",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v8":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v8")
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v8_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v8",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v10-1_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v10-1",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v11-1_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v11-1",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v11-2_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v11-2",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    elif MODEL_TO_USE == "splade_based_custom_trained_combined_references_v11-3_position_penalty+length_penalty":
        return splade_based_keyphrase_extraction(doc, top_k = top_k, model_name="custom_trained_combined_references_v11-3",
                                                 apply_position_penalty=True, length_penalty = -0.25)
    
    elif MODEL_TO_USE == "ukg_v1":
        return ukg_keyphrase_generation(doc, top_k = top_k, 
                                        phraseness_model_name = "phraseness_500k_v2",
                                        informativeness_model_name = "custom_trained_combined_references_v8")
    elif MODEL_TO_USE == "ukg_v1_position_penalty+length_penalty":
        return ukg_keyphrase_generation(doc, top_k = top_k, 
                                        phraseness_model_name = "phraseness_500k_v2",
                                        informativeness_model_name = "custom_trained_combined_references_v8",
                                        apply_position_penalty=True, length_penalty = -0.25)
    
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(doc.lower(), top_k = top_k, 
                                                        informativeness_model_name="custom_trained_combined_references_v6",
                                                        apply_position_penalty=True, length_penalty=-0.25)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_v6-2_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(doc.lower(), top_k = top_k, 
                                                        informativeness_model_name="custom_trained_combined_references_v6-2",
                                                        apply_position_penalty=True, length_penalty=-0.25)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_v11-2_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(doc.lower(), top_k = top_k, 
                                                        informativeness_model_name="custom_trained_combined_references_v11-2",
                                                        apply_position_penalty=True, length_penalty=-0.25)
    
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(doc.lower(), top_k = top_k, 
                                                        informativeness_model_name="custom_trained_combined_references_nounphrase_v6-1",
                                                        apply_position_penalty=True, length_penalty=-0.25)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-2_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(doc.lower(), top_k = top_k, 
                                                        informativeness_model_name="custom_trained_combined_references_nounphrase_v6-2",
                                                        apply_position_penalty=True, length_penalty=-0.25)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-3_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(doc.lower(), top_k = top_k, 
                                                        informativeness_model_name="custom_trained_combined_references_nounphrase_v6-3",
                                                        apply_position_penalty=True, length_penalty=-0.25)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-4_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(doc.lower(), top_k = top_k, 
                                                        informativeness_model_name="custom_trained_combined_references_nounphrase_v6-4",
                                                        apply_position_penalty=True, length_penalty=-0.25)
    elif MODEL_TO_USE == "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-5_position_penalty+length_penalty":
        return retrieval_based_ukg_keyphrase_generation(doc.lower(), top_k = top_k, 
                                                        informativeness_model_name="custom_trained_combined_references_nounphrase_v6-5",
                                                        apply_position_penalty=True, length_penalty=-0.25)
    

    elif MODEL_TO_USE == "uokg-1":
        alpha_by_dataset = {"semeval": -1,"inspec": -1,"nus": -0.75,"krapivin": -0.25}
        return keyphrase_generation_batch_uokg(docs = [doc], alpha = alpha_by_dataset.get(DATASET_TO_USE, -0.5), model_run_index=1)
    elif MODEL_TO_USE == "uokg-2":
        alpha_by_dataset = {"semeval": -1,"inspec": -1,"nus": -0.75,"krapivin": -0.5}
        return keyphrase_generation_batch_uokg(docs = [doc], alpha = alpha_by_dataset.get(DATASET_TO_USE, -0.5), model_run_index=2)
    elif MODEL_TO_USE == "uokg-3":
        alpha_by_dataset = {"semeval": -0.5,"inspec": -1,"nus": -0.25,"krapivin": -0.25}
        return keyphrase_generation_batch_uokg(docs = [doc], alpha = alpha_by_dataset.get(DATASET_TO_USE, -0.5), model_run_index=3)
    elif MODEL_TO_USE == "uokg-4":
        alpha_by_dataset = {"semeval": -1,"inspec": -0.75,"nus": -0.25,"krapivin": -0.25}
        return keyphrase_generation_batch_uokg(docs = [doc], alpha = alpha_by_dataset.get(DATASET_TO_USE, -0.5), model_run_index=4)
    elif MODEL_TO_USE == "uokg-5":
        alpha_by_dataset = {"semeval": -1,"inspec": -1,"nus": -0.5,"krapivin": -0.5}
        return keyphrase_generation_batch_uokg(docs = [doc], alpha = alpha_by_dataset.get(DATASET_TO_USE, -0.5), model_run_index=5)
    

    elif MODEL_TO_USE == "autokeygen-1":
        from src.autokeygen import keyphrase_generation_batch as keyphrase_generation_batch_autokeygen
        return keyphrase_generation_batch_autokeygen(docs = [doc], alpha = 0, model_run_index=1)
    elif MODEL_TO_USE == "autokeygen-2":
        from src.autokeygen import keyphrase_generation_batch as keyphrase_generation_batch_autokeygen
        return keyphrase_generation_batch_autokeygen(docs = [doc], alpha = 0, model_run_index=2)
    elif MODEL_TO_USE == "autokeygen-3":
        from src.autokeygen import keyphrase_generation_batch as keyphrase_generation_batch_autokeygen
        return keyphrase_generation_batch_autokeygen(docs = [doc], alpha = 0, model_run_index=3)
    elif MODEL_TO_USE == "autokeygen-4":
        from src.autokeygen import keyphrase_generation_batch as keyphrase_generation_batch_autokeygen
        return keyphrase_generation_batch_autokeygen(docs = [doc], alpha = 0, model_run_index=4)
    elif MODEL_TO_USE == "autokeygen-5":
        from src.autokeygen import keyphrase_generation_batch as keyphrase_generation_batch_autokeygen
        return keyphrase_generation_batch_autokeygen(docs = [doc], alpha = 0, model_run_index=5)
    

    elif MODEL_TO_USE == "copyrnn-1":
        from src.copyrnn import keyphrase_generation_batch as keyphrase_generation_batch_copyrnn
        return keyphrase_generation_batch_copyrnn(docs = [doc], top_k = top_k, alpha = 0, model_run_index=1)
    elif MODEL_TO_USE == "copyrnn-2":
        from src.copyrnn import keyphrase_generation_batch as keyphrase_generation_batch_copyrnn
        return keyphrase_generation_batch_copyrnn(docs = [doc], top_k = top_k, alpha = 0, model_run_index=2)
    elif MODEL_TO_USE == "copyrnn-3":    
        from src.copyrnn import keyphrase_generation_batch as keyphrase_generation_batch_copyrnn
        return keyphrase_generation_batch_copyrnn(docs = [doc], top_k = top_k, alpha = 0, model_run_index=3)
    elif MODEL_TO_USE == "copyrnn-4":
        from src.copyrnn import keyphrase_generation_batch as keyphrase_generation_batch_copyrnn    
        return keyphrase_generation_batch_copyrnn(docs = [doc], top_k = top_k, alpha = 0, model_run_index=4)
    elif MODEL_TO_USE == "copyrnn-5":
        from src.copyrnn import keyphrase_generation_batch as keyphrase_generation_batch_copyrnn
        return keyphrase_generation_batch_copyrnn(docs = [doc], top_k = top_k, alpha = 0, model_run_index=5)
    
    elif MODEL_TO_USE == "nounphrase_extraction_1_5":
        return nounphrase_extraction_as_keyphrase_generation(doc, [1, 5])
    elif MODEL_TO_USE == "nounphrase_extraction_2_5":
        return nounphrase_extraction_as_keyphrase_generation(doc, [2, 5])
    

    elif MODEL_TO_USE == "tpg-1":
        from src.tpg import tpg_keyphrase_generation as tpg_keyphrase_generation
        return tpg_keyphrase_generation(doc, top_k = top_k, model_run_index=1)
    elif MODEL_TO_USE == "tpg-2":
        from src.tpg import tpg_keyphrase_generation as tpg_keyphrase_generation
        return tpg_keyphrase_generation(doc, top_k = top_k, model_run_index=2)
    elif MODEL_TO_USE == "tpg-3":
        from src.tpg import tpg_keyphrase_generation as tpg_keyphrase_generation
        return tpg_keyphrase_generation(doc, top_k = top_k, model_run_index=3)
    elif MODEL_TO_USE == "tpg-4":
        from src.tpg import tpg_keyphrase_generation as tpg_keyphrase_generation
        return tpg_keyphrase_generation(doc, top_k = top_k, model_run_index=4)
    elif MODEL_TO_USE == "tpg-5":
        from src.tpg import tpg_keyphrase_generation as tpg_keyphrase_generation
        return tpg_keyphrase_generation(doc, top_k = top_k, model_run_index=5)
    else:
        raise NotImplementedError

# get entire dataset
dataset = process_dataset(dataset_name=DATASET_TO_USE)


processed_dataset = []
for sample in tqdm(dataset):
    document = sample.get("text_not_lowered")
    present_keyphrases = sample.get("present_keyphrases")
    absent_keyphrases = sample.get("absent_keyphrases")

    if isinstance(document, str):
        automatically_extracted_keyphrases = do_keyphrase_extraction(document, top_k = 50)
        automatically_extracted_keyphrases = {
            "present_keyphrases":  [item[0] for item in automatically_extracted_keyphrases["present"]],
            "absent_keyphrases":  [item[0] for item in automatically_extracted_keyphrases["absent"]],
        }
    else: 
        print(type(document))
        automatically_extracted_keyphrases = {
            "present_keyphrases": [],
            "absent_keyphrases": []
        }

    line = {
        "document": document,
        "present_keyphrases": present_keyphrases,
        "absent_keyphrases": absent_keyphrases,
        "automatically_extracted_keyphrases": automatically_extracted_keyphrases,
    }

    # if DATASET_TO_USE in RETRIEVAL_DATASETS:
    line.pop("document", None)


    processed_dataset.append(line)

with open(RESULT_FILE, "w") as f:
    json.dump(processed_dataset, f, indent = 4)