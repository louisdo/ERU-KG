embeddings_folder="/home/abodke2/kpg/ERU-KG/classification/scirepeval_classification/embeddings"
classification_folder="/home/abodke2/kpg/ERU-KG/classification/scirepeval_classification"
# dataset_name="scirepeval_fos_test"
dataset_name="scirepeval_mesh_descriptors_test"
# also change in .py file!!!

# CUDA_VISIBLE_DEVICES=0 \
# EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}.json" \
# DATASET_PATH="${classification_folder}/${dataset_name}/${dataset_name}.json" \
# KEYPHRASE_EXPANSION=0 \
# python custom_generate_embeddings.py


CUDA_VISIBLE_DEVICES=0 \
EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_copyrnn-1.json" \
DATASET_PATH="${classification_folder}/${dataset_name}/${dataset_name}_keyphrase_expansion_copyrnn-1.json" \
python custom_generate_embeddings.py


CUDA_VISIBLE_DEVICES=0 \
EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_autokeygen-1.json" \
DATASET_PATH="${classification_folder}/${dataset_name}/${dataset_name}_keyphrase_expansion_autokeygen-1.json" \
python custom_generate_embeddings.py


for i in $(seq 2 3);
do
    CUDA_VISIBLE_DEVICES=0 \
    EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_uokg-${i}.json" \
    DATASET_PATH="${classification_folder}/${dataset_name}/${dataset_name}_keyphrase_expansion_uokg-${i}.json" \
    python custom_generate_embeddings.py


    CUDA_VISIBLE_DEVICES=0 \
    EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    DATASET_PATH="${classification_folder}/${dataset_name}/${dataset_name}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    python custom_generate_embeddings.py


    CUDA_VISIBLE_DEVICES=0 \
    EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_copyrnn-${i}.json" \
    DATASET_PATH="${classification_folder}/${dataset_name}/${dataset_name}_keyphrase_expansion_copyrnn-${i}.json" \
    python custom_generate_embeddings.py


    CUDA_VISIBLE_DEVICES=0 \
    EMBEDDINGS_SAVE_PATH="${embeddings_folder}/embeddings--specter2_base--${dataset_name}_keyphrase_expansion_autokeygen-${i}.json" \
    DATASET_PATH="${classification_folder}/${dataset_name}/${dataset_name}_keyphrase_expansion_autokeygen-${i}.json" \
    python custom_generate_embeddings.py
done