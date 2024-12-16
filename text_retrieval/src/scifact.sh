# for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact" \
EXPERIMENT_NAME="scifact" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
python bm25_search_v3.py


# # for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_embedrank_sent2vec" \
# EXPERIMENT_NAME="scifact_keyphrase_expansion_embedrank_sent2vec" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
# python bm25_search_v3.py

# for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_doct5queries" \
EXPERIMENT_NAME="scifact_doct5queries" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
python bm25_search_v3.py


INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_doct5queries" \
EXPERIMENT_NAME="scifact_doct5queries_rm3" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
USE_RM3=1 \
python bm25_search_v3.py


#----------------------------------------

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact" \
EXPERIMENT_NAME="scifact_uokg-1 [query expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scifact_queries--uokg-1.json" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_uokg-1" \
EXPERIMENT_NAME="scifact_uokg-1 [doc expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_only_present_keyphrase_expansion_uokg-1" \
EXPERIMENT_NAME="scifact_uokg-1 [doc present keyphrase expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_uokg-1" \
EXPERIMENT_NAME="scifact_uokg-1 [query + doc expansion]" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scifact_queries--uokg-1.json" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
python bm25_search_v3.py

#----------------------------------------

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact" \
EXPERIMENT_NAME="scifact_autokeygen-1 [query expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scifact_queries--autokeygen-1.json" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_autokeygen-1" \
EXPERIMENT_NAME="scifact_autokeygen-1 [doc expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_only_present_keyphrase_expansion_autokeygen-1" \
EXPERIMENT_NAME="scifact_autokeygen-1 [doc present keyphrase expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_autokeygen-1" \
EXPERIMENT_NAME="scifact_autokeygen-1 [query + doc expansion]" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scifact_queries--autokeygen-1.json" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
python bm25_search_v3.py

#----------------------------------------

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact" \
EXPERIMENT_NAME="scifact_copyrnn-1 [query expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scifact_queries--copyrnn-1.json" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_copyrnn-1" \
EXPERIMENT_NAME="scifact_copyrnn-1 [doc expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_only_present_keyphrase_expansion_copyrnn-1" \
EXPERIMENT_NAME="scifact_copyrnn-1 [doc present keyphrase expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_copyrnn-1" \
EXPERIMENT_NAME="scifact_copyrnn-1 [query + doc expansion]" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scifact_queries--copyrnn-1.json" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
python bm25_search_v3.py

#----------------------------------------

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact" \
EXPERIMENT_NAME="scifact_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty [query expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scifact_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
EXPERIMENT_NAME="scifact_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty [doc expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
EXPERIMENT_NAME="scifact_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty [doc present keyphrase expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/scifact_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
EXPERIMENT_NAME="scifact_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty [query + doc expansion]" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="scifact" \
QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scifact_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" \
python bm25_search_v3.py

#----------------------------------------

