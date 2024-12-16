from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher('/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/index/nq320k_keyphrase_expansion_splade_based_splade-cocondenser-ensembledistil')
hits = searcher.search('when was the writ watch invented by who', k = 50)


print(hits[0].lucene_document.get('raw'))
for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')