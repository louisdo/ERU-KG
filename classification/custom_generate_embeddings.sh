# embeddings_folder="/scratch/lamdo/scirepeval_classification/embeddings/"
embeddings_folder="/home/abodke2/kpg/ERU-KG/classification/arxiv_classification/embeddings_only_present"

classification_folder="/home/abodke2/kpg/ERU-KG/classification/arxiv_classification"

folder_name='arxiv_classification_only_present'
# folder_name='arxiv_classification'
# folder_name='scirepeval_fos_test'
# folder_name='scirepeval_mesh_descriptors_test'

# dataset_name="scirepeval_fos_test"
# dataset_name="scirepeval_mesh_descriptors_test"
dataset_name="arxiv_classification"

# also change in .py file!!!

CUDA_VISIBLE_DEVICES=0 \
EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}.json" \
DATASET_PATH="${classification_folder}/${dataset_name}/${dataset_name}.json" \
KEYPHRASE_EXPANSION=0 \
python custom_generate_embeddings.py

for i in $(seq 1 3);
do
    CUDA_VISIBLE_DEVICES=0 \
    EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_uokg-${i}.json" \
    DATASET_PATH="${classification_folder}/${folder_name}/${dataset_name}_keyphrase_expansion_uokg-${i}.json" \
    python custom_generate_embeddings.py


    CUDA_VISIBLE_DEVICES=0 \
    EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    DATASET_PATH="${classification_folder}/${folder_name}/${dataset_name}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    python custom_generate_embeddings.py


    CUDA_VISIBLE_DEVICES=0 \
    EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_copyrnn-${i}.json" \
    DATASET_PATH="${classification_folder}/${folder_name}/${dataset_name}_keyphrase_expansion_copyrnn-${i}.json" \
    python custom_generate_embeddings.py


    CUDA_VISIBLE_DEVICES=0 \
    EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_autokeygen-${i}.json" \
    DATASET_PATH="${classification_folder}/${folder_name}/${dataset_name}_keyphrase_expansion_autokeygen-${i}.json" \
    python custom_generate_embeddings.py
done