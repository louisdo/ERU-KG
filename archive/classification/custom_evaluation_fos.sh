embeddings_folder="/scratch/lamdo/scirepeval_classification/embeddings"
dataset_name="scirepeval_fos_test"
output_file="experiments/scirepeval_fos.jsonl"


# no expansion
EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--scirepeval_fos_test.json" \
EXPERIMENT_NAME="no_expansion" \
OUTPUT_FILE=$output_file \
python custom_evaluation_fos.py

for i in $(seq 1 3);
do
    # EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    # EXPERIMENT_NAME="eru-kg-${i}" \
    # OUTPUT_FILE=$output_file \
    # python custom_evaluation_fos.py

    # EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_copyrnn-${i}.json" \
    # EXPERIMENT_NAME="copyrnn-${i}" \
    # OUTPUT_FILE=$output_file \
    # python custom_evaluation_fos.py

    # EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_autokeygen-${i}.json" \
    # EXPERIMENT_NAME="autokeygen-${i}" \
    # OUTPUT_FILE=$output_file \
    # python custom_evaluation_fos.py

    # EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_uokg-${i}.json" \
    # EXPERIMENT_NAME="uokg-${i}" \
    # OUTPUT_FILE=$output_file \
    # python custom_evaluation_fos.py

    EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty.json" \
    EXPERIMENT_NAME="eru-kg-small-${i}" \
    OUTPUT_FILE=$output_file \
    python custom_evaluation_fos.py
done



ablations=("no_queries" "no_cc" "no_titles")
for ablation in "${ablations[@]}"; do
    EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty.json" \
    EXPERIMENT_NAME="eru-kg-small-${i}" \
    OUTPUT_FILE=$output_file \
    python custom_evaluation_fos.py
done