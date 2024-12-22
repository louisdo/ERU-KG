classification_folder="/scratch/lamdo/scirepeval_classification/"
dataset_name="scirepeval_fos_test"

mkdir "${classification_folder}/${dataset_name}"

# no expansion
DATASET_NAME="${dataset_name}" \
OUTPUT_FILE="${classification_folder}/${dataset_name}/${dataset_name}.json" \
python prepare_data_for_evaluation.py


for i in $(seq 1 3);
do
    KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset_name}--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    DATASET_NAME="${dataset_name}" \
    OUTPUT_FILE="${classification_folder}/${dataset_name}/scirepeval_fos_test_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    python prepare_data_for_evaluation.py



    KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset_name}--autokeygen-${i}.json" \
    DATASET_NAME="${dataset_name}" \
    OUTPUT_FILE="${classification_folder}/${dataset_name}/scirepeval_fos_test_keyphrase_expansion_autokeygen-${i}.json" \
    python prepare_data_for_evaluation.py

    KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset_name}--copyrnn-${i}.json" \
    DATASET_NAME="${dataset_name}" \
    OUTPUT_FILE="${classification_folder}/${dataset_name}/scirepeval_fos_test_keyphrase_expansion_copyrnn-${i}.json" \
    python prepare_data_for_evaluation.py

    KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset_name}--uokg-${i}.json" \
    DATASET_NAME="${dataset_name}" \
    OUTPUT_FILE="${classification_folder}/${dataset_name}/scirepeval_fos_test_keyphrase_expansion_uokg-${i}.json" \
    python prepare_data_for_evaluation.py
done