EMBEDDING_FILE="/scratch/lamdo/scirepeval_classification/embeddings/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" \
EXPERIMENT_NAME="eru-kg-1" \
python custom_evaluation_fos.py


EMBEDDING_FILE="/scratch/lamdo/scirepeval_classification/embeddings/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_copyrnn-1.json" \
EXPERIMENT_NAME="copyrnn-1" \
python custom_evaluation_fos.py


EMBEDDING_FILE="/scratch/lamdo/scirepeval_classification/embeddings/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_autokeygen-1.json" \
EXPERIMENT_NAME="autokeygen-1" \
python custom_evaluation_fos.py

EMBEDDING_FILE="/scratch/lamdo/scirepeval_classification/embeddings/embeddings--specter2_base--scirepeval_fos_test_keyphrase_expansion_uokg-1.json" \
EXPERIMENT_NAME="uokg-1" \
python custom_evaluation_fos.py


EMBEDDING_FILE="/scratch/lamdo/scirepeval_classification/embeddings/embeddings--specter2_base--scirepeval_fos_test.json" \
EXPERIMENT_NAME="no_expansion" \
python custom_evaluation_fos.py