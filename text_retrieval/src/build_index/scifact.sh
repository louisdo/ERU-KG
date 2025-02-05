# OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection" \
# INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact" \
# python scifact.py


# OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection_doct5queries" \
# KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scifact--doct5queries.json" \
# INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact_doct5queries" \
# python scifact.py


# for i in $(seq 1 3);
# do
    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection_uokg-${i}" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scifact--uokg-${i}.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_uokg-${i}" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python scifact.py


    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection_autokeygen-${i}" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scifact--autokeygen-${i}.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_autokeygen-${i}" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python scifact.py

    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection_copyrnn-${i}" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scifact--copyrnn-${i}.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_copyrnn-${i}" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python scifact.py


    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scifact--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python scifact.py

    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scifact--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python scifact.py


    #------------------------------------------------------------------


    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection_uokg-${i}" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scifact--uokg-${i}.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact_only_present_keyphrase_expansion_uokg-${i}" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python scifact.py


    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection_autokeygen-${i}" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scifact--autokeygen-${i}.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact_only_present_keyphrase_expansion_autokeygen-${i}" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python scifact.py

    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection_copyrnn-${i}" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scifact--copyrnn-${i}.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact_only_present_keyphrase_expansion_copyrnn-${i}" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python scifact.py


    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scifact--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python scifact.py


    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scifact--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python scifact.py

    # echo "input${i}_now"
# done

OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_10" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scifact--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_10.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_10" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python scifact.py

OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/scifact_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_50" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scifact--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_50.json" \
INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_50" \
EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
python scifact.py