# eru-kg
models_types=(
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty"
    "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-2_position_penalty+length_penalty" 
    "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-3_position_penalty+length_penalty"
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-4_position_penalty+length_penalty"
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-5_position_penalty+length_penalty"
)
datasets=("scirepeval_fos_test" "scirepeval_mesh_descriptors_test")
result_folder="/scratch/lamdo/precompute_keyphrase_extraction/"

for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"

        CUDA_VISIBLE_DEVICES=2 DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
    done
done

# eru-kg-small
models_types=(
    "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-1_position_penalty+length_penalty"
    "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-2_position_penalty+length_penalty" 
    "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-3_position_penalty+length_penalty"
)
datasets=("scirepeval_fos_test" "scirepeval_mesh_descriptors_test")
result_folder="/scratch/lamdo/precompute_keyphrase_extraction/"

for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"

        CUDA_VISIBLE_DEVICES=1 DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
    done
done


# autokeygen
models_types=(
    "autokeygen-2"
    "autokeygen-3"
)
datasets=("scirepeval_fos_test" "scirepeval_mesh_descriptors_test")
result_folder="/scratch/lamdo/precompute_keyphrase_extraction/"

for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"

        CUDA_VISIBLE_DEVICES=1 DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction.py
    done
done


# copyrnn
models_types=(
    "copyrnn-2"
    "copyrnn-3"
)
datasets=("scirepeval_fos_test" "scirepeval_mesh_descriptors_test")
result_folder="/scratch/lamdo/precompute_keyphrase_extraction/"

for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"

        CUDA_VISIBLE_DEVICES=2 DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction.py
    done
done

# uokg
models_types=(
    "uokg-2"
    "uokg-3"
)
datasets=("scirepeval_fos_test" "scirepeval_mesh_descriptors_test")
result_folder="/scratch/lamdo/precompute_keyphrase_extraction/"

for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"

        CUDA_VISIBLE_DEVICES=0 DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction.py
    done
done