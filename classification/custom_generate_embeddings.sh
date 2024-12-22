CUDA_VISIBLE_DEVICES=2 \
EMBEDDINGS_SAVE_PATH="/scratch/lamdo/scirepeval_classification/embeddings/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" \
DATASET_PATH="/scratch/lamdo/scirepeval_classification/fos/scirepeval_fos_test_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" \
python custom_generate_embeddings.py


CUDA_VISIBLE_DEVICES=2 \
EMBEDDINGS_SAVE_PATH="/scratch/lamdo/scirepeval_classification/embeddings/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_uokg-1.json" \
DATASET_PATH="/scratch/lamdo/scirepeval_classification/fos/scirepeval_fos_test_keyphrase_expansion_uokg-1.json" \
python custom_generate_embeddings.py


CUDA_VISIBLE_DEVICES=2 \
EMBEDDINGS_SAVE_PATH="/scratch/lamdo/scirepeval_classification/embeddings/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_copyrnn-1.json" \
DATASET_PATH="/scratch/lamdo/scirepeval_classification/fos/scirepeval_fos_test_keyphrase_expansion_copyrnn-1.json" \
python custom_generate_embeddings.py


CUDA_VISIBLE_DEVICES=0 \
EMBEDDINGS_SAVE_PATH="/scratch/lamdo/scirepeval_classification/embeddings/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_autokeygen-1.json" \
DATASET_PATH="/scratch/lamdo/scirepeval_classification/fos/scirepeval_fos_test_keyphrase_expansion_autokeygen-1.json" \
python custom_generate_embeddings.py

CUDA_VISIBLE_DEVICES=1 \
EMBEDDINGS_SAVE_PATH="/scratch/lamdo/scirepeval_classification/embeddings/embeddings--specter2_base--scirepeval_fos_test.json" \
DATASET_PATH="/scratch/lamdo/scirepeval_classification/fos/scirepeval_fos_test_keyphrase_expansion_autokeygen-1.json" \
KEYPHRASE_EXPANSION=0 \
python custom_generate_embeddings.py