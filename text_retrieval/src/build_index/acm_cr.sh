# OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/acm_cr_collections/pyserini_formatted_collection" \
# INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/acm_cr" \
# python acm_cr.py

for i in $(seq 1 3);
do
    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/acm_cr_collections/pyserini_formatted_collection_uokg-${i}" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/acm_cr--uokg-${i}.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/acm_cr_keyphrase_expansion_uokg-${i}" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python acm_cr.py


    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/acm_cr_collections/pyserini_formatted_collection_autokeygen-${i}" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/acm_cr--autokeygen-${i}.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/acm_cr_keyphrase_expansion_autokeygen-${i}" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python acm_cr.py

    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/acm_cr_collections/pyserini_formatted_collection_copyrnn-${i}" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/acm_cr--copyrnn-${i}.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/acm_cr_keyphrase_expansion_copyrnn-${i}" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python acm_cr.py


    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/acm_cr_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/acm_cr--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/acm_cr_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python acm_cr.py

    OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/acm_cr_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty" \
    KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/acm_cr--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty.json" \
    INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/acm_cr_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
    NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    python acm_cr.py


    #------------------------------------------------------------------


    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/acm_cr_collections/pyserini_formatted_collection_uokg-${i}" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/acm_cr--uokg-${i}.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/acm_cr_only_present_keyphrase_expansion_uokg-${i}" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python acm_cr.py


    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/acm_cr_collections/pyserini_formatted_collection_autokeygen-${i}" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/acm_cr--autokeygen-${i}.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/acm_cr_only_present_keyphrase_expansion_autokeygen-${i}" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python acm_cr.py

    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/acm_cr_collections/pyserini_formatted_collection_copyrnn-${i}" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/acm_cr--copyrnn-${i}.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/acm_cr_only_present_keyphrase_expansion_copyrnn-${i}" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python acm_cr.py


    # OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/acm_cr_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    # KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/acm_cr--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    # INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/acm_cr_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    # EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    # NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    # python acm_cr.py


    OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/acm_cr_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty" \
    KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/acm_cr--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty.json" \
    INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/acm_cr_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    python acm_cr.py

    # echo "input${i}_now"
done
