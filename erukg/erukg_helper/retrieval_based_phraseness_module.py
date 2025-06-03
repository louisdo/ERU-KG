from collections import Counter
from tqdm import tqdm
import traceback
import json, os, time

from erukg.nounphrase_extractor import CandidateExtractorRegExpNLTK
from erukg.erukg_helper.config import GENERAL_CONFIG
from erukg.erukg_helper.utils import maybe_create_folder

class DocumentRetriever:
    def __init__(self, index_path):
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(index_path)
        self.index_path = index_path

        self.searcher.set_bm25(k1=0.9, b=0.4)


    def search_single_query(self,
                            query, 
                            top_k = 100, 
                            return_raw = False, 
                            # bm25 = {"k1": 0.9, "b": 0.4}, 
                            fields={"contents": 1.0}):
        if not query: return []
        
        hits = self.searcher.search(query, k = top_k, fields=fields)
        # hits = self.searcher.batch_search([query], qids = [str(_) for _ in list(range(1))], k=top_k, threads=4, fields=fields).get("0", [])
        # print(hits)

        res = []
        for line in hits:
            docid = line.docid
            score = line.score

            new_line = {
                "docid": docid,
                "score": score
            }
            if return_raw:
                raw = line.lucene_document.get('raw')
                new_line["raw"] = raw

            res.append(new_line)

        return res
    
    def batch_search(self,
                     queries,
                     top_k=100,
                     return_raw=False,
                     fields={"contents": 1.0}):
        if not queries:
            return []

        # self.searcher.set_bm25(k1=bm25["k1"], b=bm25["b"])

        # Use Pyserini's batch_search method
        batch_results = self.searcher.batch_search(queries, qids = [str(_) for _ in list(range(len(queries)))], k=top_k, threads=16, fields=fields)
        
        processed_results = []
        for index in range(len(queries)):
            str_index = str(index)
            hits = batch_results.get(str_index, [])

            query_processed = []
            for line in hits:
                docid = line.docid
                score = line.score

                result = {
                    "docid": docid,
                    "score": score
                }
                if return_raw:
                    raw = line.lucene_document.get('raw')
                    result["raw"] = raw
                query_processed.append(result)
            processed_results.append(query_processed)

        return processed_results


class RetrievalBasedPhrasenessModule:
    def __init__(self, 
                 document_index_path: str, 
                 neighbor_size: int,
                 document_index_download_url: str = None,
                 beta: float = 0.75,
                 document_index_phrase_field: str = "present_keyphrases",
                 document_index_vector_field: str = "vector",
                 informativeness_model_name: str = "",
                 no_retrieval = False):
        self.document_index_path = document_index_path
        self.document_index_download_url = document_index_download_url
        self.no_retrieval = no_retrieval

        self.doc_retriever = None

        if not no_retrieval:
            self.load_retriever()
        else:
            self.doc_retriever = None
        self.candext = CandidateExtractorRegExpNLTK([1, 5])
        # self.candext_2_5 = CandidateExtractorRegExpNLTK([2,5])

        self.document_index_phrase_field = document_index_phrase_field
        self.document_index_vector_field = document_index_vector_field

        assert 0 <= beta <= 1
        self._set_beta(beta)
        self._set_neighbor_size(neighbor_size)

        cache_dir = GENERAL_CONFIG["cache_dir"]
        phrase_vocab_cache_dir = os.path.join(cache_dir, "phrase_vocab")
        maybe_create_folder(phrase_vocab_cache_dir)
        # self.cache_path = f"/scratch/lamdo/erukg_cache_{informativeness_model_name}.json"
        self.cache_path = os.path.join(phrase_vocab_cache_dir, f"erukg_cache_{informativeness_model_name}.json")

        self.phrase_glossary, self.docid2phraseid, self.docid2tokenscore = self._build_phrase_glossary()

    def load_retriever(self):
        if os.path.exists(self.document_index_path):
            self.doc_retriever = DocumentRetriever(index_path=self.document_index_path)
        else:
            message = f"Index not found at '{self.document_index_path}'."
            if self.document_index_download_url is not None:
                message += f"\nPlease download the index from '{self.document_index_download_url}', unzip it, and place the index in '{self.document_index_path}'"
            raise FileNotFoundError(message)

    def _set_beta(self, beta):
        self.beta = beta
        self.one_minus_beta = 1 - beta

    def _set_neighbor_size(self, neighbor_size):
        self.neighbor_size = neighbor_size

    def _set_no_retrieval(self, no_retrieval: bool):
        self.no_retrieval = no_retrieval
        if no_retrieval is False:
            if self.doc_retriever is None: self.load_retriever()


    def _build_phrase_glossary(self, cutoff = 3):
        if self.doc_retriever is None or self.no_retrieval is True: return [], {}, {}

        if not os.path.exists(self.cache_path):
            phrase_counter = Counter()
            docid2phrase = {}
            docid2tokenscore = {}
            for i in tqdm(range(self.doc_retriever.searcher.num_docs), desc = "Building phrase glossary"):
                doc = self.doc_retriever.searcher.doc(i)
                docid = doc.docid()

                docraw = json.loads(doc.raw())
                # print(doc.docid(), doc.contents()) 
                phrase_counter.update(docraw[self.document_index_phrase_field])
                docid2phrase[docid] = docraw[self.document_index_phrase_field]
                docid2tokenscore[docid] = docraw[self.document_index_vector_field]

            # cutoff
            phrase_counter = Counter({k:v for k,v in phrase_counter.items() if v >= cutoff})
            phrase_vocab = list(sorted(phrase_counter.keys(), key = lambda x: phrase_counter[x], reverse = True))
            phrase2id = {phrase:i for i,phrase in enumerate(phrase_vocab)}

            docid2phraseid = {k: [phrase2id[phrase] for phrase in v if phrase in phrase_counter] for k,v in docid2phrase.items()}
            # save for next time usage
            to_save = {
                "phrase_vocab": phrase_vocab,
                "docid2phraseid": docid2phraseid,
                "docid2tokenscore": docid2tokenscore
            }
            with open(self.cache_path, "w") as f:
                json.dump(to_save, f)
        else:
            with open(self.cache_path) as f:
                cache_data = json.load(f)
                phrase_vocab = cache_data["phrase_vocab"]
                docid2phraseid = cache_data["docid2phraseid"]
                docid2tokenscore = cache_data["docid2tokenscore"]

        print(f"Number of phrases in the glossary {len(phrase_vocab)}")
        return phrase_vocab, docid2phraseid, docid2tokenscore

    def _extract(self, doc):
        return self.candext(doc)
    
    def _extract_batch(self, docs):
        return [self.candext(doc) for doc in docs]
    
    def _retrieve(self, doc, return_pruning_latency = False):
        if self.doc_retriever is None or self.no_retrieval is True:
            if return_pruning_latency:
                return Counter(), None, None, None
            return Counter(), None, None
        try:
            if self.beta == 1: return Counter(), None, None
            # return Counter()
            retrieval_results = self.doc_retriever.search_single_query(
                query = doc, 
                top_k = self.neighbor_size,
                return_raw=False
            )

            start = time.time()
            scores = [item.get("score") for item in retrieval_results]
            docids = [item.get("docid") for item in retrieval_results]
            total_scores = sum(scores)
            scores = [score / total_scores for score in scores]

            # raws = [json.loads(item.get("raw")) for item in retrieval_results]
            # retrieved_documents_vectors = [item.get(self.document_index_vector_field) for item in raws]
            retrieved_documents_vectors = [self.docid2tokenscore[docid] for docid in docids]
            # print(raws[0])

            all_nounphrases = Counter()
            for docid, doc_score in zip(docids, scores):
                nounphrases = [self.phrase_glossary[j] for j in self.docid2phraseid[docid]]

                if nounphrases: 
                    _temp = doc_score / len(nounphrases)
                    for nounp in nounphrases:
                        all_nounphrases[nounp] += _temp
            res = Counter(dict(all_nounphrases.most_common(100)))
            end = time.time()
            # return all_nounphrases
            if return_pruning_latency:
                return res, retrieved_documents_vectors, scores, end - start
            return res, retrieved_documents_vectors, scores
            # return all_nounphrases, retrieved_documents_vectors
        except Exception as e:
            print("Error in phraseness module retrieval section", e)
            if return_pruning_latency:
                return Counter(), None, None, None
            return Counter(), None, None
        
    def _retrieve_batch(self, docs):
        if self.doc_retriever is None or self.no_retrieval is True:
            return [Counter() for _ in docs], [None for _ in docs], [None for _ in docs]
        try:
            if self.beta == 1: return [Counter() for _ in docs], [None for _ in docs], [None for _ in docs]
            # Perform batch search
            batch_results = self.doc_retriever.batch_search(
                queries=docs,
                top_k=self.neighbor_size,
                return_raw=False
            )

            all_doc_nounphrases = []
            all_retrieved_documents_vectors = []
            all_scores = []

            for retrieval_results in batch_results:
                scores = [item.get("score") for item in retrieval_results]
                docids = [item.get("docid") for item in retrieval_results]
                total_scores = sum(scores)
                normalized_scores = [score / total_scores for score in scores]
                all_scores.append(normalized_scores)

                # raws = [json.loads(item.get("raw")) for item in retrieval_results]
                # retrieved_documents_vectors = [item.get(self.document_index_vector_field) for item in raws]
                retrieved_documents_vectors = [self.docid2tokenscore[docid] for docid in docids]

                doc_nounphrases = Counter()
                for docid, doc_score in zip(docids, normalized_scores):
                    nounphrases = [self.phrase_glossary[j] for j in self.docid2phraseid[docid]]
                    if nounphrases:
                        for nounp in nounphrases:
                            doc_nounphrases[nounp] += doc_score / len(nounphrases)

                all_doc_nounphrases.append(Counter(dict(doc_nounphrases.most_common(100))))
                all_retrieved_documents_vectors.append(retrieved_documents_vectors)

            return all_doc_nounphrases, all_retrieved_documents_vectors, all_scores

        except Exception as e:
            print(f"Error in batch retrieval: {traceback.format_exc()}")
            return [Counter() for _ in docs], [None for _ in docs], [None for _ in docs]
        

    def _combined_retrieval_and_extraction(self, extracted_nounphrases, retrieved_nounphrases):
        extracted_nounphrases_counter = Counter(extracted_nounphrases)
        retrieved_nounphrases_counter = retrieved_nounphrases
        
        norm_extracted = sum(extracted_nounphrases_counter.values())
        extracted_nounphrases_counter = Counter({k:v / norm_extracted for k, v in extracted_nounphrases_counter.items()})

        norm_retrieved = sum(retrieved_nounphrases_counter.values())
        retrieved_nounphrases_counter = Counter({k:v / norm_retrieved for k, v in retrieved_nounphrases_counter.items()})

        all_extracted_phrases = set(extracted_nounphrases_counter.keys())
        all_retrieved_phrases = set(retrieved_nounphrases_counter.keys())

        all_phrases = all_extracted_phrases.union(all_retrieved_phrases)

        res = Counter({phrase: self.beta * extracted_nounphrases_counter[phrase] + self.one_minus_beta * retrieved_nounphrases_counter[phrase] for phrase in all_phrases})

        return res
    


    def __call__(self, doc, return_retrieved_documents_vectors = False, return_retrieved_documents_scores = False, return_pruning_latency = False):
        extracted_nounphrases = self._extract(doc)

        if return_pruning_latency:
            retrieved_nounphrases, retrieved_documents_vectors, retrieved_documents_scores, pruning_latency = self._retrieve(doc, return_pruning_latency=True)
        else:
            retrieved_nounphrases, retrieved_documents_vectors, retrieved_documents_scores = self._retrieve(doc)
            pruning_latency = None

        res = {
            "keyphrase_candidates_scores": self._combined_retrieval_and_extraction(
                extracted_nounphrases=extracted_nounphrases,
                retrieved_nounphrases=retrieved_nounphrases
            ),
            "pruning_latency": pruning_latency
        }

        if return_retrieved_documents_vectors:
            res["retrieved_documents_vectors"] = retrieved_documents_vectors

        if return_retrieved_documents_scores:
            res["retrieved_documents_scores"] = retrieved_documents_scores
        
        return res
    

    def batch_generation(self, 
                         docs, 
                         return_retrieved_documents_vectors = False,
                         return_retrieved_documents_scores = False):
        batch_extracted_nounphrases = self._extract_batch(docs)
        batch_retrieved_nounphrases, batch_retrieved_documents_vectors, batch_retrieved_documents_scores = self._retrieve_batch(docs)

        res = {
            "keyphrase_candidates_scores": [self._combined_retrieval_and_extraction(extracted_nounphrases, retrieved_nounphrases) \
                    for extracted_nounphrases, retrieved_nounphrases in zip (batch_extracted_nounphrases, batch_retrieved_nounphrases)]
        }


        if return_retrieved_documents_vectors:
            res["retrieved_documents_vectors"] = batch_retrieved_documents_vectors
        
        if return_retrieved_documents_scores:
            res["retrieved_documents_scores"] = batch_retrieved_documents_scores

        return res
    