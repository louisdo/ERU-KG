# OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/nfcorpus_collections/pyserini_formatted_collection" \
# INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/nfcorpus" \
# python nfcorpus.py


# OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/nfcorpus_collections/pyserini_formatted_collection_doct5queries" \
# KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus--doct5queries.json" \
# INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/nfcorpus_doct5queries" \
# python nfcorpus.py


OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/nfcorpus_collections/pyserini_formatted_collection_uokg-1" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus--uokg-1.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_uokg-1" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python nfcorpus.py


OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/nfcorpus_collections/pyserini_formatted_collection_autokeygen-1" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus--autokeygen-1.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_autokeygen-1" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python nfcorpus.py

OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/nfcorpus_collections/pyserini_formatted_collection_copyrnn-1" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus--copyrnn-1.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_copyrnn-1" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python nfcorpus.py


OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/nfcorpus_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python nfcorpus.py


#------------------------------------------------------------------


OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/nfcorpus_collections/pyserini_formatted_collection_uokg-1" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus--uokg-1.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/nfcorpus_only_present_keyphrase_expansion_uokg-1" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python nfcorpus.py


OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/nfcorpus_collections/pyserini_formatted_collection_autokeygen-1" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus--autokeygen-1.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/nfcorpus_only_present_keyphrase_expansion_autokeygen-1" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python nfcorpus.py

OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/nfcorpus_collections/pyserini_formatted_collection_copyrnn-1" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus--copyrnn-1.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/nfcorpus_only_present_keyphrase_expansion_copyrnn-1" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python nfcorpus.py


OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/nfcorpus_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/nfcorpus_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python nfcorpus.py