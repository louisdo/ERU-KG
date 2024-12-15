# # for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_validation_evaluation" \
# EXPERIMENT_NAME="scirepeval_search_validation_evaluation" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search" \
# python bm25_search_v3.py 


# # for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_validation_evaluation_keyphrase_expansion_embedrank_sent2vec" \
# EXPERIMENT_NAME="scirepeval_search_validation_evaluation_keyphrase_expansion_embedrank_sent2vec" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search" \
# python bm25_search_v3.py 


# # for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_validation_evaluation_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty" \
# EXPERIMENT_NAME="scirepeval_search_validation_evaluation_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search" \
# python bm25_search_v3.py


# for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scirepeval_search_validation_evaluation_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty" \
EXPERIMENT_NAME="scirepeval_search_validation_evaluation_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty [with query expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scirepeval_search" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_validation_evaluation_queries--retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty.json" \
python bm25_search_v3.py