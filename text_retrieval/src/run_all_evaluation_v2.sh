datasets=(
    scidocs
    scifact
    trec_covid
    nfcorpus
    doris_mae
    acm_cr
)

for dataset in "${datasets[@]}"; do
    for i in $(seq 1 1);
    do

        # ----------------------------------------

        INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/${dataset}" \
        EXPERIMENT_NAME="${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty [query expansion]" \
        GROUNDTRUTH_DATA_PATH="" DATASET_NAME="${dataset}" \
        QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset}_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty.json" \
        python bm25_search_v3.py

        # INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/${dataset}" \
        # EXPERIMENT_NAME="${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty [query present keyphrase expansion]" \
        # GROUNDTRUTH_DATA_PATH="" DATASET_NAME="${dataset}" \
        # QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset}_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty.json" \
        # EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
        # python bm25_search_v3.py

        INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty" \
        EXPERIMENT_NAME="${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty [doc expansion]" \
        GROUNDTRUTH_DATA_PATH="" DATASET_NAME="${dataset}" \
        python bm25_search_v3.py

        # INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/${dataset}_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty" \
        # EXPERIMENT_NAME="${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty [doc present keyphrase expansion]" \
        # GROUNDTRUTH_DATA_PATH="" DATASET_NAME="${dataset}" \
        # python bm25_search_v3.py

        INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty" \
        EXPERIMENT_NAME="${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty [query + doc expansion]" \
        GROUNDTRUTH_DATA_PATH="" DATASET_NAME="${dataset}" \
        QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset}_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-${i}_position_penalty+length_penalty.json" \
        python bm25_search_v3.py

        # ----------------------------------------

        INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/${dataset}" \
        EXPERIMENT_NAME="${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty [query expansion]" \
        GROUNDTRUTH_DATA_PATH="" DATASET_NAME="${dataset}" \
        QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset}_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty.json" \
        python bm25_search_v3.py

        # INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/${dataset}" \
        # EXPERIMENT_NAME="${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty [query present keyphrase expansion]" \
        # GROUNDTRUTH_DATA_PATH="" DATASET_NAME="${dataset}" \
        # QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset}_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty.json" \
        # EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
        # python bm25_search_v3.py

        INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty" \
        EXPERIMENT_NAME="${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty [doc expansion]" \
        GROUNDTRUTH_DATA_PATH="" DATASET_NAME="${dataset}" \
        python bm25_search_v3.py

        # INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/${dataset}_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty" \
        # EXPERIMENT_NAME="${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty [doc present keyphrase expansion]" \
        # GROUNDTRUTH_DATA_PATH="" DATASET_NAME="${dataset}" \
        # python bm25_search_v3.py

        INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty" \
        EXPERIMENT_NAME="${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty [query + doc expansion]" \
        GROUNDTRUTH_DATA_PATH="" DATASET_NAME="${dataset}" \
        QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset}_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-${i}_position_penalty+length_penalty.json" \
        python bm25_search_v3.py

    done
done