# # for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs" \
# EXPERIMENT_NAME="scidocs" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
# python bm25_search_v3.py

# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs" \
# EXPERIMENT_NAME="scidocs_rm3" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
# USE_RM3=1 \
# python bm25_search_v3.py

# # for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_doct5queries" \
# EXPERIMENT_NAME="scidocs_doct5queries" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
# python bm25_search_v3.py


# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_doct5queries" \
# EXPERIMENT_NAME="scidocs_doct5queries_rm3" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
# USE_RM3=1 \
# python bm25_search_v3.py


for i in $(seq 1 1);
do
    #----------------------------------------

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs" \
    EXPERIMENT_NAME="scidocs_uokg-${i} [query expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scidocs_queries--uokg-${i}.json" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs" \
    EXPERIMENT_NAME="scidocs_uokg-${i} [query present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scidocs_queries--uokg-${i}.json" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_keyphrase_expansion_uokg-${i}" \
    EXPERIMENT_NAME="scidocs_uokg-${i} [doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_only_present_keyphrase_expansion_uokg-${i}" \
    EXPERIMENT_NAME="scidocs_uokg-${i} [doc present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_keyphrase_expansion_uokg-${i}" \
    EXPERIMENT_NAME="scidocs_uokg-${i} [query + doc expansion]" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scidocs_queries--uokg-${i}.json" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    python bm25_search_v3.py

    #----------------------------------------

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs" \
    EXPERIMENT_NAME="scidocs_autokeygen-${i} [query expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scidocs_queries--autokeygen-${i}.json" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs" \
    EXPERIMENT_NAME="scidocs_autokeygen-${i} [query present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scidocs_queries--autokeygen-${i}.json" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_keyphrase_expansion_autokeygen-${i}" \
    EXPERIMENT_NAME="scidocs_autokeygen-${i} [doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_only_present_keyphrase_expansion_autokeygen-${i}" \
    EXPERIMENT_NAME="scidocs_autokeygen-${i} [doc present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_keyphrase_expansion_autokeygen-${i}" \
    EXPERIMENT_NAME="scidocs_autokeygen-${i} [query + doc expansion]" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scidocs_queries--autokeygen-${i}.json" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    python bm25_search_v3.py

    #----------------------------------------

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs" \
    EXPERIMENT_NAME="scidocs_copyrnn-${i} [query expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scidocs_queries--copyrnn-${i}.json" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs" \
    EXPERIMENT_NAME="scidocs_copyrnn-${i} [query present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scidocs_queries--copyrnn-${i}.json" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_keyphrase_expansion_copyrnn-${i}" \
    EXPERIMENT_NAME="scidocs_copyrnn-${i} [doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_only_present_keyphrase_expansion_copyrnn-${i}" \
    EXPERIMENT_NAME="scidocs_copyrnn-${i} [doc present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_keyphrase_expansion_copyrnn-${i}" \
    EXPERIMENT_NAME="scidocs_copyrnn-${i} [query + doc expansion]" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scidocs_queries--copyrnn-${i}.json" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    python bm25_search_v3.py

    #----------------------------------------

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs" \
    EXPERIMENT_NAME="scidocs_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [query expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scidocs_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs" \
    EXPERIMENT_NAME="scidocs_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [query present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scidocs_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    EXPERIMENT_NAME="scidocs_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    EXPERIMENT_NAME="scidocs_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [doc present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scidocs_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    EXPERIMENT_NAME="scidocs_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [query + doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scidocs" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scidocs_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    python bm25_search_v3.py

    #----------------------------------------
done