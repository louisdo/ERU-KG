# for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation" \
EXPERIMENT_NAME="scirepeval_search_evaluation" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
python bm25_search_v3.py


# # for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation_keyphrase_expansion_embedrank_sent2vec" \
# EXPERIMENT_NAME="scirepeval_search_evaluation_keyphrase_expansion_embedrank_sent2vec" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
# python bm25_search_v3.py

# # for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation_doct5queries" \
# EXPERIMENT_NAME="scirepeval_search_evaluation_doct5queries" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
# python bm25_search_v3.py


# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation_doct5queries" \
# EXPERIMENT_NAME="scirepeval_search_evaluation_doct5queries_rm3" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
# USE_RM3=1 \
# python bm25_search_v3.py




# for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation_keyphrase_expansion_uokg-1" \
EXPERIMENT_NAME="scirepeval_search_evaluation_uokg-1 [with query expansion]" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_evaluation_queries--uokg-1.json" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
python bm25_search_v3.py


INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation_keyphrase_expansion_autokeygen-1" \
EXPERIMENT_NAME="scirepeval_search_evaluation_autokeygen-1 [with query expansion]" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_evaluation_queries--autokeygen-1.json" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
python bm25_search_v3.py


INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation_keyphrase_expansion_copyrnn-1" \
EXPERIMENT_NAME="scirepeval_search_evaluation_copyrnn-1 [with query expansion]" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_evaluation_queries--copyrnn-1.json" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
python bm25_search_v3.py



INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation_keyphrase_expansion_uokg-1" \
EXPERIMENT_NAME="scirepeval_search_evaluation_uokg-1" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
python bm25_search_v3.py


INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation_keyphrase_expansion_autokeygen-1" \
EXPERIMENT_NAME="scirepeval_search_evaluation_autokeygen-1" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
python bm25_search_v3.py


INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation_keyphrase_expansion_copyrnn-1" \
EXPERIMENT_NAME="scirepeval_search_evaluation_copyrnn-1" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
python bm25_search_v3.py



INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
EXPERIMENT_NAME="scirepeval_search_evaluation_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
python bm25_search_v3.py


INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
EXPERIMENT_NAME="scirepeval_search_evaluation_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty [with query expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_evaluation_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" \
python bm25_search_v3.py



# ------

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation" \
EXPERIMENT_NAME="scirepeval_search_evaluation_uokg-1 [only query expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_evaluation_queries--uokg-1.json" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation" \
EXPERIMENT_NAME="scirepeval_search_evaluation_autokeygen-1 [only query expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_evaluation_queries--autokeygen-1.json" \
python bm25_search_v3.py


INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation" \
EXPERIMENT_NAME="scirepeval_search_evaluation_copyrnn-1 [only query expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_evaluation_queries--copyrnn-1.json" \
python bm25_search_v3.py


INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_evaluation" \
EXPERIMENT_NAME="scirepeval_search_evaluation_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty [only query expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search_evaluation" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_evaluation_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" \
python bm25_search_v3.py