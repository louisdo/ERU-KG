device_to_use=0

# eru-kg
models_types=(
    "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty"
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-2_position_penalty+length_penalty" 
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-3_position_penalty+length_penalty"
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-4_position_penalty+length_penalty"
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-5_position_penalty+length_penalty"
)
# datasets=("scirepeval_fos_test" "scirepeval_mesh_descriptors_test")
# datasets=(
#     "arxiv_classification"
#     "scirepeval_fos_test"
# )
datasets=(
    "arxiv_classification_title"
    # "scirepeval_fos_test_title"
)
result_folder="/scratch/lamdo/precompute_keyphrase_extraction/"

for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"

        CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
    done
done

# eru-kg-small
models_types=(
    "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-1_position_penalty+length_penalty"
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-2_position_penalty+length_penalty" 
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-3_position_penalty+length_penalty"
)
result_folder="/scratch/lamdo/precompute_keyphrase_extraction/"

for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"

        CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
    done
done



# eru-kg-ablations
models_types=(
    "retrieval_based_ukg_custom_trained_combined_references_no_titles_nounphrase_v6-1_position_penalty+length_penalty"
    # "retrieval_based_ukg_custom_trained_combined_references_no_queries_nounphrase_v6-1_position_penalty+length_penalty" 
    # "retrieval_based_ukg_custom_trained_combined_references_no_cc_nounphrase_v6-1_position_penalty+length_penalty"
)
result_folder="/scratch/lamdo/precompute_keyphrase_extraction/"

for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"

        CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
    done
done



# autokeygen
models_types=(
    "autokeygen-1"
    # "autokeygen-2"
    # "autokeygen-3"
)
result_folder="/scratch/lamdo/precompute_keyphrase_extraction/"

for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"

        CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction.py
    done
done


# copyrnn
models_types=(
    "copyrnn-1"
    # "copyrnn-2"
    # "copyrnn-3"
)
result_folder="/scratch/lamdo/precompute_keyphrase_extraction/"

for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"

        CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction.py
    done
done

# uokg
models_types=(
    "uokg-1"
    # "uokg-2"
    # "uokg-3"
)
result_folder="/scratch/lamdo/precompute_keyphrase_extraction/"

for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"

        CUDA_VISIBLE_DEVICES=1 DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction.py
    done
done