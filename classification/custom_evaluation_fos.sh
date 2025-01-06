embeddings_folder="/home/abodke2/kpg/ERU-KG/classification/scirepeval_classification/embeddings"
dataset_name="scirepeval_fos_test"


# no expansion
EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--scirepeval_fos_test.json" \
EXPERIMENT_NAME="no_expansion" \
python custom_evaluation_fos.py

for i in $(seq 1 3);
do
    EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    EXPERIMENT_NAME="eru-kg-${i}" \
    python custom_evaluation_fos.py

    EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_copyrnn-${i}.json" \
    EXPERIMENT_NAME="copyrnn-${i}" \
    python custom_evaluation_fos.py

    EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_autokeygen-${i}.json" \
    EXPERIMENT_NAME="autokeygen-${i}" \
    python custom_evaluation_fos.py

    EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_uokg-${i}.json" \
    EXPERIMENT_NAME="uokg-${i}" \
    python custom_evaluation_fos.py
done

