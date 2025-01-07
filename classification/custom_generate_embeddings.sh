# embeddings_folder="/scratch/lamdo/scirepeval_classification/embeddings/"
embeddings_folder="/scratch/lamdo/arxiv_classification/embeddings"

classification_folder="/scratch/lamdo/arxiv_classification/"
# classification_folder="/scratch/lamdo/scirepeval_classification/"

# folder_name='arxiv_classification_only_present'
folder_name='arxiv_classification'
# folder_name='scirepeval_fos_test'
# folder_name='scirepeval_mesh_descriptors_test'

# dataset_name="scirepeval_fos_test"
# dataset_name="scirepeval_mesh_descriptors_test"
dataset_name="arxiv_classification"

cuda_device=0

# also change in .py file!!!

# CUDA_VISIBLE_DEVICES=$cuda_device \
# EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}.json" \
# DATASET_PATH="${classification_folder}/${dataset_name}/${dataset_name}.json" \
# KEYPHRASE_EXPANSION=0 \
# python custom_generate_embeddings.py

# for i in $(seq 1 3);
# do
#     # CUDA_VISIBLE_DEVICES=$cuda_device \
#     # EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_uokg-${i}.json" \
#     # DATASET_PATH="${classification_folder}/${folder_name}/${dataset_name}_keyphrase_expansion_uokg-${i}.json" \
#     # python custom_generate_embeddings.py


#     # CUDA_VISIBLE_DEVICES=$cuda_device \
#     # EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
#     # DATASET_PATH="${classification_folder}/${folder_name}/${dataset_name}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
#     # python custom_generate_embeddings.py


#     # CUDA_VISIBLE_DEVICES=$cuda_device \
#     # EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_copyrnn-${i}.json" \
#     # DATASET_PATH="${classification_folder}/${folder_name}/${dataset_name}_keyphrase_expansion_copyrnn-${i}.json" \
#     # python custom_generate_embeddings.py


#     # CUDA_VISIBLE_DEVICES=$cuda_device \
#     # EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_autokeygen-${i}.json" \
#     # DATASET_PATH="${classification_folder}/${folder_name}/${dataset_name}_keyphrase_expansion_autokeygen-${i}.json" \
#     # python custom_generate_embeddings.py

#     CUDA_VISIBLE_DEVICES=$cuda_device \
#     EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty.json" \
#     DATASET_PATH="${classification_folder}/${folder_name}/${dataset_name}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-${i}_position_penalty+length_penalty.json" \
#     python custom_generate_embeddings.py
# done


# ablations

ablations=("no_queries" "no_cc" "no_titles")
for ablation in "${ablations[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda_device \
    EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty.json" \
    DATASET_PATH="${classification_folder}/${folder_name}/${dataset_name}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_${ablation}_nounphrase_v6-1_position_penalty+length_penalty.json" \
    python custom_generate_embeddings.py
done