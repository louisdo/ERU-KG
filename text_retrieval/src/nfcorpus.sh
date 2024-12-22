# for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus" \
EXPERIMENT_NAME="nfcorpus" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
python bm25_search_v3.py

INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus" \
EXPERIMENT_NAME="nfcorpus_rm3" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
USE_RM3=1 \
python bm25_search_v3.py


# # for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
# INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_embedrank_sent2vec" \
# EXPERIMENT_NAME="nfcorpus_keyphrase_expansion_embedrank_sent2vec" \
# GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
# python bm25_search_v3.py

# for scirepeval dataset, no need to specify GROUNDTRUTH_DATA_PATH since we will load it from datasets package
INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_doct5queries" \
EXPERIMENT_NAME="nfcorpus_doct5queries" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
python bm25_search_v3.py


INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_doct5queries" \
EXPERIMENT_NAME="nfcorpus_doct5queries_rm3" \
GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
USE_RM3=1 \
python bm25_search_v3.py


for i in $(seq 1 3);
do
    #----------------------------------------

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus" \
    EXPERIMENT_NAME="nfcorpus_uokg-${i} [query expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus_queries--uokg-${i}.json" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus" \
    EXPERIMENT_NAME="nfcorpus_uokg-${i} [query present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus_queries--uokg-${i}.json" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_uokg-${i}" \
    EXPERIMENT_NAME="nfcorpus_uokg-${i} [doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_only_present_keyphrase_expansion_uokg-${i}" \
    EXPERIMENT_NAME="nfcorpus_uokg-${i} [doc present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_uokg-${i}" \
    EXPERIMENT_NAME="nfcorpus_uokg-${i} [query + doc expansion]" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus_queries--uokg-${i}.json" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    python bm25_search_v3.py

    #----------------------------------------

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus" \
    EXPERIMENT_NAME="nfcorpus_autokeygen-${i} [query expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus_queries--autokeygen-${i}.json" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus" \
    EXPERIMENT_NAME="nfcorpus_autokeygen-${i} [query present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus_queries--autokeygen-${i}.json" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_autokeygen-${i}" \
    EXPERIMENT_NAME="nfcorpus_autokeygen-${i} [doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_only_present_keyphrase_expansion_autokeygen-${i}" \
    EXPERIMENT_NAME="nfcorpus_autokeygen-${i} [doc present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_autokeygen-${i}" \
    EXPERIMENT_NAME="nfcorpus_autokeygen-${i} [query + doc expansion]" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus_queries--autokeygen-${i}.json" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    python bm25_search_v3.py

    #----------------------------------------

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus" \
    EXPERIMENT_NAME="nfcorpus_copyrnn-${i} [query expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus_queries--copyrnn-${i}.json" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus" \
    EXPERIMENT_NAME="nfcorpus_copyrnn-${i} [query present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus_queries--copyrnn-${i}.json" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_copyrnn-${i}" \
    EXPERIMENT_NAME="nfcorpus_copyrnn-${i} [doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_only_present_keyphrase_expansion_copyrnn-${i}" \
    EXPERIMENT_NAME="nfcorpus_copyrnn-${i} [doc present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_copyrnn-${i}" \
    EXPERIMENT_NAME="nfcorpus_copyrnn-${i} [query + doc expansion]" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus_queries--copyrnn-${i}.json" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    python bm25_search_v3.py

    #----------------------------------------

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus" \
    EXPERIMENT_NAME="nfcorpus_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [query expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus" \
    EXPERIMENT_NAME="nfcorpus_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [query present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    EXPANSION_ONLY_PRESENT_KEYPHRASES=1 \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    EXPERIMENT_NAME="nfcorpus_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_only_present_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    EXPERIMENT_NAME="nfcorpus_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [doc present keyphrase expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    python bm25_search_v3.py

    INDEX_PATH="/scratch/lamdo/pyserini_experiments/index/nfcorpus_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" \
    EXPERIMENT_NAME="nfcorpus_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty [query + doc expansion]" \
    GROUNDTRUTH_DATA_PATH="" DATASET_NAME="nfcorpus" \
    QUERY_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/nfcorpus_queries--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    python bm25_search_v3.py

    #----------------------------------------
done