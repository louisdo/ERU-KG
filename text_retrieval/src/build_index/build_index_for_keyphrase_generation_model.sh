# INPUT_FILE="/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search_validation_evaluation.jsonl"  \
# OUTPUT_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/collections/scirepeval_collections/splade_based_custom_trained_combined_references_v6-1_position_penalty+length_penalty" \
# INDEX_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v6-1" \
# KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_validation_evaluation--nounphrase_extraction_1_5.json" \
# DOCUMENT_VECTORS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v6-1--scirepeval_search_validation_evaluation.jsonl" \
# python scirepeval_search_v2.py



# INPUT_FILE="/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search_validation_evaluation.jsonl"  \
# OUTPUT_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/collections/scirepeval_collections/splade_based_custom_trained_combined_references_v6-2_position_penalty+length_penalty" \
# INDEX_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v6-2" \
# KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_validation_evaluation--nounphrase_extraction_1_5.json" \
# DOCUMENT_VECTORS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v6-2--scirepeval_search_validation_evaluation.jsonl" \
# python scirepeval_search_v2.py


# INPUT_FILE="/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search_validation_evaluation.jsonl"  \
# OUTPUT_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/collections/scirepeval_collections/splade_based_custom_trained_combined_references_v6-3_position_penalty+length_penalty" \
# INDEX_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v6-3" \
# KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_validation_evaluation--nounphrase_extraction_1_5.json" \
# DOCUMENT_VECTORS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v6-3--scirepeval_search_validation_evaluation.jsonl" \
# python scirepeval_search_v2.py



# INPUT_FILE="/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search_validation_evaluation.jsonl"  \
# OUTPUT_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/collections/scirepeval_collections/splade_based_custom_trained_combined_references_v6-4_position_penalty+length_penalty" \
# INDEX_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v6-4" \
# KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_validation_evaluation--nounphrase_extraction_1_5.json" \
# DOCUMENT_VECTORS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v6-4--scirepeval_search_validation_evaluation.jsonl" \
# python scirepeval_search_v2.py


# INPUT_FILE="/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search_validation_evaluation.jsonl"  \
# OUTPUT_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/collections/scirepeval_collections/splade_based_custom_trained_combined_references_v6-5_position_penalty+length_penalty" \
# INDEX_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v6-5" \
# KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_validation_evaluation--nounphrase_extraction_1_5.json" \
# DOCUMENT_VECTORS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v6-5--scirepeval_search_validation_evaluation.jsonl" \
# python scirepeval_search_v2.py


# for i in $(seq 1 5);
# do
#     INPUT_FILE="/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search_validation_evaluation.jsonl"  \
#     OUTPUT_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/collections/scirepeval_collections/splade_based_custom_trained_combined_references_v6-${i}_position_penalty+length_penalty" \
#     INDEX_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v6-${i}" \
#     KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_validation_evaluation--nounphrase_extraction_1_5.json" \
#     DOCUMENT_VECTORS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v6-${i}--scirepeval_search_validation_evaluation.jsonl" \
#     python scirepeval_search_v2.py
# done


for i in $(seq 1 3);
do
    INPUT_FILE="/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search_validation_evaluation.jsonl"  \
    OUTPUT_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/collections/scirepeval_collections/splade_based_custom_trained_combined_references_v7-${i}_position_penalty+length_penalty" \
    INDEX_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_v7-${i}" \
    KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_validation_evaluation--nounphrase_extraction_1_5.json" \
    DOCUMENT_VECTORS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v7-${i}--scirepeval_search_validation_evaluation.jsonl" \
    python scirepeval_search_v2.py
done


ablations=("no_queries" "no_cc" "no_titles")

for name in "${ablations[@]}";
do
    INPUT_FILE="/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search_validation_evaluation.jsonl"  \
    OUTPUT_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/collections/scirepeval_collections/splade_based_custom_trained_combined_references_${name}_v6-1_position_penalty+length_penalty" \
    INDEX_FOLDER="/scratch/lamdo/keyphrase_generation_retrieval_index/index/scirepeval_search_validation_evaluation_nounphrase_${name}_v6-1" \
    KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_search_validation_evaluation--nounphrase_extraction_1_5.json" \
    DOCUMENT_VECTORS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_${name}_v6-1--scirepeval_search_validation_evaluation.jsonl" \
    python scirepeval_search_v2.py
done