DATASET_NAME="scirepeval_fos_test" \
OUTPUT_FILE="/scratch/lamdo/scirepeval_classification/fos/scirepeval_fos_test.json" \
python prepare_data_for_evaluation.py


KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_fos_test--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" \
DATASET_NAME="scirepeval_fos_test" \
OUTPUT_FILE="/scratch/lamdo/scirepeval_classification/fos/scirepeval_fos_test_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json" \
python prepare_data_for_evaluation.py



KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_fos_test--autokeygen-1.json" \
DATASET_NAME="scirepeval_fos_test" \
OUTPUT_FILE="/scratch/lamdo/scirepeval_classification/fos/scirepeval_fos_test_keyphrase_expansion_autokeygen-1.json" \
python prepare_data_for_evaluation.py

KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_fos_test--copyrnn-1.json" \
DATASET_NAME="scirepeval_fos_test" \
OUTPUT_FILE="/scratch/lamdo/scirepeval_classification/fos/scirepeval_fos_test_keyphrase_expansion_copyrnn-1.json" \
python prepare_data_for_evaluation.py

KEYPHRASE_EXPANSION_PATH="/scratch/lamdo/precompute_keyphrase_extraction/scirepeval_fos_test--uokg-1.json" \
DATASET_NAME="scirepeval_fos_test" \
OUTPUT_FILE="/scratch/lamdo/scirepeval_classification/fos/scirepeval_fos_test_keyphrase_expansion_uokg-1.json" \
python prepare_data_for_evaluation.py