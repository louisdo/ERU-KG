datasets=(
    scifact
    scidocs
    trec_covid
    nfcorpus
    doris_mae
    acm_cr
)

for dataset in "${datasets[@]}"; do
    OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/${dataset}_collections/pyserini_formatted_collection" \
    INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/${dataset}" \
    python ${dataset}.py

    # for i in $(seq 1 3);
    # do

    #     OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/${dataset}_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty" \
    #     KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/${dataset}--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty.json" \
    #     INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty" \
    #     EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
    #     NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    #     python ${dataset}.py

    #     OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/${dataset}_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty" \
    #     KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/${dataset}--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty.json" \
    #     INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty" \
    #     EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
    #     NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    #     python ${dataset}.py


    #     # ------------------------------------------------------------------


    #     OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/${dataset}_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty" \
    #     KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/${dataset}--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty.json" \
    #     INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/${dataset}_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty" \
    #     EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    #     NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    #     python ${dataset}.py


    #     OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/${dataset}_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty" \
    #     KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/${dataset}--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty.json" \
    #     INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/${dataset}_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty" \
    #     EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    #     NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
    #     python ${dataset}.py

    #     echo "input${i}_now"
    # done

done