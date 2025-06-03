device_to_use=0
datasets=(
    "semeval" 
    "inspec" 
    "nus" 
    "krapivin" 
    "kp20k" 
)
result_folder="/scratch/lamdo/keyphrase_generation_results/results_ongoing/"

# eru-kg-hardneg (final ver)
models_types=(
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-1_position_penalty+length_penalty"
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-2_position_penalty+length_penalty" 
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-3_position_penalty+length_penalty"

    # "eru_kg_1_alpha_1_beta_1"
    # "eru_kg_1_alpha_0.8_beta_0.8"
    # "eru_kg_1_alpha_0.6_beta_0.6"
    # "eru_kg_1_alpha_0.4_beta_0.4"
    # "eru_kg_1_alpha_0.2_beta_0.2"
    # "eru_kg_1_alpha_0_beta_0"

    "eru_kg_2_alpha_1_beta_1"
    "eru_kg_2_alpha_0.8_beta_0.8"
    "eru_kg_2_alpha_0.6_beta_0.6"
    "eru_kg_2_alpha_0.4_beta_0.4"
    "eru_kg_2_alpha_0.2_beta_0.2"
    "eru_kg_2_alpha_0_beta_0"

    "eru_kg_3_alpha_1_beta_1"
    "eru_kg_3_alpha_0.8_beta_0.8"
    "eru_kg_3_alpha_0.6_beta_0.6"
    "eru_kg_3_alpha_0.4_beta_0.4"
    "eru_kg_3_alpha_0.2_beta_0.2"
    "eru_kg_3_alpha_0_beta_0"
    # "eru_kg_2_alpha_0.8_beta_0.8"
    # "eru_kg_3_alpha_0.8_beta_0.8"
)
for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"

        CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
    done
done

# # eru-kg (neighbor size != 100) (final ver)
# models_types=(
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-1_position_penalty+length_penalty_neighborsize_10"
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-1_position_penalty+length_penalty_neighborsize_50"
# )

# for dataset in "${datasets[@]}"; do
#     for model_type in "${models_types[@]}"; do
#         echo "Config: $dataset - $model_type - $top_k"

#         CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
#     done
# done


# # eru-kg (neighbor size != 100)
# models_types=(
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_50"
# )

# for dataset in "${datasets[@]}"; do
#     for model_type in "${models_types[@]}"; do
#         echo "Config: $dataset - $model_type - $top_k"

#         CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
#     done
# done

# # eru-kg
# models_types=(
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty"
#     # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-2_position_penalty+length_penalty" 
#     # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-3_position_penalty+length_penalty"
#     # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-4_position_penalty+length_penalty"
#     # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-5_position_penalty+length_penalty"
# )

# for dataset in "${datasets[@]}"; do
#     for model_type in "${models_types[@]}"; do
#         echo "Config: $dataset - $model_type - $top_k"

#         CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
#     done
# done

# # eru-kg-small
# models_types=(
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-1_position_penalty+length_penalty"
#     # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-2_position_penalty+length_penalty" 
#     # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-3_position_penalty+length_penalty"
# )

# for dataset in "${datasets[@]}"; do
#     for model_type in "${models_types[@]}"; do
#         echo "Config: $dataset - $model_type - $top_k"

#         CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
#     done
# done



# # eru-kg-ablations
# models_types=(
#     "retrieval_based_ukg_custom_trained_combined_references_no_titles_nounphrase_v6-1_position_penalty+length_penalty"
#     # "retrieval_based_ukg_custom_trained_combined_references_no_queries_nounphrase_v6-1_position_penalty+length_penalty" 
#     # "retrieval_based_ukg_custom_trained_combined_references_no_cc_nounphrase_v6-1_position_penalty+length_penalty"
# )

# for dataset in "${datasets[@]}"; do
#     for model_type in "${models_types[@]}"; do
#         echo "Config: $dataset - $model_type - $top_k"

#         CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
#     done
# done

# # eru-kg-ablations (final ver)
# models_types=(
#     "retrieval_based_ukg_custom_trained_combined_references_no_titles_nounphrase_v8-1_position_penalty+length_penalty"
#     "retrieval_based_ukg_custom_trained_combined_references_no_queries_nounphrase_v8-1_position_penalty+length_penalty" 
#     "retrieval_based_ukg_custom_trained_combined_references_no_cc_nounphrase_v8-1_position_penalty+length_penalty"
# )

# for dataset in "${datasets[@]}"; do
#     for model_type in "${models_types[@]}"; do
#         echo "Config: $dataset - $model_type - $top_k"

#         CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
#     done
# done


# # eru-kg-small (final ver)
# models_types=(
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-1_position_penalty+length_penalty"
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-2_position_penalty+length_penalty" 
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v9-3_position_penalty+length_penalty"
# )

# for dataset in "${datasets[@]}"; do
#     for model_type in "${models_types[@]}"; do
#         echo "Config: $dataset - $model_type - $top_k"

#         CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
#     done
# done



# # autokeygen
# models_types=(
#     "autokeygen-1"
#     # "autokeygen-2"
#     # "autokeygen-3"
# )

# for dataset in "${datasets[@]}"; do
#     for model_type in "${models_types[@]}"; do
#         echo "Config: $dataset - $model_type - $top_k"

#         CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction.py
#     done
# done


# # copyrnn
# models_types=(
#     "copyrnn-1"
#     # "copyrnn-2"
#     # "copyrnn-3"
# )

# for dataset in "${datasets[@]}"; do
#     for model_type in "${models_types[@]}"; do
#         echo "Config: $dataset - $model_type - $top_k"

#         CUDA_VISIBLE_DEVICES=$device_to_use DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction.py
#     done
# done

# # uokg
# models_types=(
#     "uokg-1"
#     # "uokg-2"
#     # "uokg-3"
# )

# for dataset in "${datasets[@]}"; do
#     for model_type in "${models_types[@]}"; do
#         echo "Config: $dataset - $model_type - $top_k"

#         CUDA_VISIBLE_DEVICES=1 DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction.py
#     done
# done

# # tpg
# datasets=(
#     "semeval" "inspec" "nus" "krapivin" "kp20k" 
#     # kp20k
# )
# result_folder="/scratch/lamdo/keyphrase_generation_results/results_ongoing/"
# models_types=(
#     "tpg-3"
# )

# for dataset in "${datasets[@]}"; do
#     for model_type in "${models_types[@]}"; do
#         echo "Config: $dataset - $model_type - $top_k"

#         CUDA_VISIBLE_DEVICES=2 DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction.py
#     done
# done


# # embedrank sbert
# datasets=(
#     "semeval" "inspec" "nus" "krapivin" "kp20k"
#     # kp20k
# )
# result_folder="/scratch/lamdo/keyphrase_generation_results/results_ongoing/"
# models_types=(
#     embedrank_sentence_transformers_all-MiniLM-L12-v2
# )

# for dataset in "${datasets[@]}"; do
#     for model_type in "${models_types[@]}"; do
#         echo "Config: $dataset - $model_type - $top_k"

#         CUDA_VISIBLE_DEVICES=2 DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction.py
#     done
# done