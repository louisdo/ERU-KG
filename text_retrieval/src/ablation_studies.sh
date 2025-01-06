for ablation in "no_titles" "no_queries" "no_cc"; do
    # scifact
    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty" \
    EXPERIMENT_NAME="scifact_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty [query + doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scifact_queries--retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty.json" \
    python bm25_search_v3.py

    # # scidocs
    # INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty" \
    # EXPERIMENT_NAME="scidocs_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty [query + doc expansion]" \
    # GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    # QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scidocs_queries--retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty.json" \
    # python bm25_search_v3.py


    # # trec_covid
    # INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/trec_covid_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty" \
    # EXPERIMENT_NAME="trec_covid_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty [query + doc expansion]" \
    # GROUNDTRUTH_DATA_PATH="" DATASET_NAME="trec_covid" \
    # QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/trec_covid_queries--retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty.json" \
    # python bm25_search_v3.py


    # nfcorpus
    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty" \
    EXPERIMENT_NAME="nfcorpus_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty [query + doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus_queries--retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty.json" \
    python bm25_search_v3.py


    # # doris_mae
    # INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty" \
    # EXPERIMENT_NAME="doris_mae_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty [query + doc expansion]" \
    # GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    # QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae_queries--retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty.json" \
    # python bm25_search_v3.py


    # # acm_cr
    # INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/acm_cr_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty" \
    # EXPERIMENT_NAME="acm_cr_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty [query + doc expansion]" \
    # GROUNDTRUTH_DATA_PATH="" DATASET_NAME="acm_cr" \
    # QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/acm_cr_queries--retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty.json" \
    # python bm25_search_v3.py
done