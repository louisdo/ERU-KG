classification_folder="/scratch/lamdo/arxiv_classification/"
dataset_name="arxiv_classification"

# classification_folder="/scratch/lamdo/scirepeval_classification/"
# dataset_name="scirepeval_fos_test"

# variation="_only_present"
variation=""
only_present=0

out_folder="${classification_folder}/${dataset_name}${variation}"

mkdir $out_folder

# no expansion
DATASET_NAME="${dataset_name}" \
OUTPUT_FILE="${out_folder}/${dataset_name}.json" \
python prepare_data_for_evaluation.py


for i in $(seq 1 3);
do
    KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset_name}--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    DATASET_NAME="${dataset_name}" \
    OUTPUT_FILE="${out_folder}/${dataset_name}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    ONLY_PRESENT_KEYPHRASES=$only_present \
    python prepare_data_for_evaluation.py



    KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset_name}--autokeygen-${i}.json" \
    DATASET_NAME="${dataset_name}" \
    OUTPUT_FILE="${out_folder}/${dataset_name}_keyphrase_expansion_autokeygen-${i}.json" \
    ONLY_PRESENT_KEYPHRASES=$only_present \
    python prepare_data_for_evaluation.py

    KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset_name}--copyrnn-${i}.json" \
    DATASET_NAME="${dataset_name}" \
    OUTPUT_FILE="${out_folder}/${dataset_name}_keyphrase_expansion_copyrnn-${i}.json" \
    ONLY_PRESENT_KEYPHRASES=$only_present \
    python prepare_data_for_evaluation.py

    # KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset_name}--uokg-${i}.json" \
    # DATASET_NAME="${dataset_name}" \
    # OUTPUT_FILE="${out_folder}/${dataset_name}_keyphrase_expansion_uokg-${i}.json" \
    # ONLY_PRESENT_KEYPHRASES=$only_present \
    # python prepare_data_for_evaluation.py


    # KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset_name}--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty.json" \
    # DATASET_NAME="${dataset_name}" \
    # OUTPUT_FILE="${out_folder}/${dataset_name}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty.json" \
    # ONLY_PRESENT_KEYPHRASES=$only_present \
    # python prepare_data_for_evaluation.py
done


ablations=("no_queries" "no_cc" "no_titles")
for ablation in "${ablations[@]}"; do
    KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/${dataset_name}--retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty.json" \
    DATASET_NAME="${dataset_name}" \
    OUTPUT_FILE="${out_folder}/${dataset_name}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty.json" \
    ONLY_PRESENT_KEYPHRASES=$only_present \
    python prepare_data_for_evaluation.py
done