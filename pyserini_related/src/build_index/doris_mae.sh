# OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/doris_mae_collections/pyserini_formatted_collection" \
# INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/doris_mae" \
# python doris_mae.py


# OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/doris_mae_collections/pyserini_formatted_collection_doct5queries" \
# KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae--doct5queries.json" \
# INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/doris_mae_doct5queries" \
# python doris_mae.py


OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/doris_mae_collections/pyserini_formatted_collection_uokg-1" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae--uokg-1.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_uokg-1" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python doris_mae.py


OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/doris_mae_collections/pyserini_formatted_collection_autokeygen-1" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae--autokeygen-1.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_autokeygen-1" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python doris_mae.py

OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/doris_mae_collections/pyserini_formatted_collection_copyrnn-1" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae--copyrnn-1.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_copyrnn-1" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python doris_mae.py


OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/doris_mae_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python doris_mae.py


#------------------------------------------------------------------


OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/doris_mae_collections/pyserini_formatted_collection_uokg-1" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae--uokg-1.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/doris_mae_only_present_keyphrase_expansion_uokg-1" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python doris_mae.py


OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/doris_mae_collections/pyserini_formatted_collection_autokeygen-1" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae--autokeygen-1.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/doris_mae_only_present_keyphrase_expansion_autokeygen-1" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python doris_mae.py

OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/doris_mae_collections/pyserini_formatted_collection_copyrnn-1" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae--copyrnn-1.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/doris_mae_only_present_keyphrase_expansion_copyrnn-1" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python doris_mae.py


OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/doris_mae_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/doris_mae_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python doris_mae.py