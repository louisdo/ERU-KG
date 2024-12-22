# for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae" \
EXPERIMENT_NAME="doris_mae" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae" \
EXPERIMENT_NAME="doris_mae_rm3" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
USE_RM3=1 \
python bm25_search_v3.py


# # for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_embedrank_sent2vec" \
# EXPERIMENT_NAME="doris_mae_keyphrase_expansion_embedrank_sent2vec" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
# python bm25_search_v3.py

# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_doct5queries" \
# EXPERIMENT_NAME="doris_mae_doct5queries" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
# python bm25_search_v3.py


# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_doct5queries" \
# EXPERIMENT_NAME="doris_mae_doct5queries_rm3" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
# USE_RM3=1 \
# python bm25_search_v3.py


for i in $(seq 1 3);
do
    #----------------------------------------

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae" \
    EXPERIMENT_NAME="doris_mae_uokg-${i} [query expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae_queries--uokg-${i}.json" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae" \
    EXPERIMENT_NAME="doris_mae_uokg-${i} [query present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae_queries--uokg-${i}.json" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_uokg-${i}" \
    EXPERIMENT_NAME="doris_mae_uokg-${i} [doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_only_present_keyphrase_expansion_uokg-${i}" \
    EXPERIMENT_NAME="doris_mae_uokg-${i} [doc present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_uokg-${i}" \
    EXPERIMENT_NAME="doris_mae_uokg-${i} [query + doc expansion]" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae_queries--uokg-${i}.json" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    python bm25_search_v3.py

    #----------------------------------------

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae" \
    EXPERIMENT_NAME="doris_mae_autokeygen-${i} [query expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae_queries--autokeygen-${i}.json" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae" \
    EXPERIMENT_NAME="doris_mae_autokeygen-${i} [query present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae_queries--autokeygen-${i}.json" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_autokeygen-${i}" \
    EXPERIMENT_NAME="doris_mae_autokeygen-${i} [doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_only_present_keyphrase_expansion_autokeygen-${i}" \
    EXPERIMENT_NAME="doris_mae_autokeygen-${i} [doc present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_autokeygen-${i}" \
    EXPERIMENT_NAME="doris_mae_autokeygen-${i} [query + doc expansion]" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae_queries--autokeygen-${i}.json" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    python bm25_search_v3.py

    #----------------------------------------

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae" \
    EXPERIMENT_NAME="doris_mae_copyrnn-${i} [query expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae_queries--copyrnn-${i}.json" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae" \
    EXPERIMENT_NAME="doris_mae_copyrnn-${i} [query present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae_queries--copyrnn-${i}.json" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_copyrnn-${i}" \
    EXPERIMENT_NAME="doris_mae_copyrnn-${i} [doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_only_present_keyphrase_expansion_copyrnn-${i}" \
    EXPERIMENT_NAME="doris_mae_copyrnn-${i} [doc present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_copyrnn-${i}" \
    EXPERIMENT_NAME="doris_mae_copyrnn-${i} [query + doc expansion]" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae_queries--copyrnn-${i}.json" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    python bm25_search_v3.py

    #----------------------------------------

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae" \
    EXPERIMENT_NAME="doris_mae_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [query expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae" \
    EXPERIMENT_NAME="doris_mae_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [query present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    EXPERIMENT_NAME="doris_mae_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    EXPERIMENT_NAME="doris_mae_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [doc present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/doris_mae_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    EXPERIMENT_NAME="doris_mae_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [query + doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="doris_mae" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/doris_mae_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    python bm25_search_v3.py

    #----------------------------------------
done