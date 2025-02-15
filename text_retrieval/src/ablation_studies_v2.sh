datasets=(
    scidocs
    scifact
    trec_covid
    nfcorpus
    doris_mae
    acm_cr
)

for dataset in "${datasets[@]}"; do
    for ablation in "no_titles" "no_queries" "no_cc"; do
        INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v8-1_position_penalty+length_penalty" \
        EXPERIMENT_NAME="${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v8-1_position_penalty+length_penalty [query + doc expansion]" \
        GROUNDTRUTH_DATA_PATH="" DATASET_NAME="${dataset}" \
        QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset}_queries--retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v8-1_position_penalty+length_penalty.json" \
        python bm25_search_v3.py
    done

    for neighborsize in 10 50; do
        INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-1_position_penalty+length_penalty_neighborsize_${neighborsize}" \
        EXPERIMENT_NAME="${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-1_position_penalty+length_penalty_neighborsize_${neighborsize} [query + doc expansion]" \
        GROUNDTRUTH_DATA_PATH="" DATASET_NAME="${dataset}" \
        QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset}_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-1_position_penalty+length_penalty_neighborsize_${neighborsize}.json" \
        python bm25_search_v3.py
    done
done