# CUDA_VISIBLE_DEVICES=1 DATASET_TO_USE="scifact" RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" MODEL_TO_USE="uke_custom_trained_combined_references_v6_position_penalty+length_penalty" python run_keyphrase_prediction_batch.py

# CUDA_VISIBLE_DEVICES=1 DATASET_TO_USE="scirepeval_search_validation_evaluation" RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" MODEL_TO_USE="uke_custom_trained_combined_references_v6_position_penalty+length_penalty" python run_keyphrase_prediction_batch.py


# CUDA_VISIBLE_DEVICES=1 \
# DATASET_TO_USE="scirepeval_search_validation_evaluation" \
# RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
# MODEL_TO_USE="uke_custom_trained_combined_references_v6_position_penalty+length_penalty" \
# PRECOMPUTED_REPRESENTATIONS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v6--scirepeval_search_validation_evaluation.jsonl" \
# python run_keyphrase_prediction_batch.py

CUDA_VISIBLE_DEVICES=1 \
DATASET_TO_USE="scirepeval_search_validation_evaluation" \
RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty" \
PRECOMPUTED_REPRESENTATIONS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v6--scirepeval_search_validation_evaluation.jsonl" \
python run_keyphrase_prediction_batch.py


CUDA_VISIBLE_DEVICES=1 \
DATASET_TO_USE="scirepeval_search_validation_evaluation_queries" \
RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty" \
python run_keyphrase_prediction_batch.py


# CUDA_VISIBLE_DEVICES=1 \
# DATASET_TO_USE="scifact" \
# RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
# MODEL_TO_USE="uke_custom_trained_combined_references_v6_position_penalty+length_penalty" \
# PRECOMPUTED_REPRESENTATIONS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v6--scifact.jsonl" \
# python run_keyphrase_prediction_batch.py


# CUDA_VISIBLE_DEVICES=0 \
# DATASET_TO_USE="scifact" \
# RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
# MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty" \
# PRECOMPUTED_REPRESENTATIONS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v6--scifact.jsonl" \
# python run_keyphrase_prediction_batch.py

# CUDA_VISIBLE_DEVICES=0 \
# DATASET_TO_USE="scifact_queries" \
# RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
# MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty" \
# PRECOMPUTED_REPRESENTATIONS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v6--scifact_queries.jsonl" \
# python run_keyphrase_prediction_batch.py

CUDA_VISIBLE_DEVICES=0 \
DATASET_TO_USE="trec_covid" \
RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty" \
python run_keyphrase_prediction_batch.py


CUDA_VISIBLE_DEVICES=2 \
DATASET_TO_USE="trec_covid_queries" \
RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty" \
python run_keyphrase_prediction_batch.py



CUDA_VISIBLE_DEVICES=0 \
DATASET_TO_USE="scidocs" \
RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty" \
python run_keyphrase_prediction_batch.py

CUDA_VISIBLE_DEVICES=0 \
DATASET_TO_USE="scidocs_queries" \
RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty" \
python run_keyphrase_prediction_batch.py


scirepeval_search_validation_evaluation_queries


CUDA_VISIBLE_DEVICES=1 \
DATASET_TO_USE="scirepeval_search_validation_evaluation" \
RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
MODEL_TO_USE="splade_based_custom_trained_combined_references_v11-2_position_penalty+length_penalty" \
PRECOMPUTED_REPRESENTATIONS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v11-2--scirepeval_search_validation_evaluation.jsonl" \
python run_keyphrase_prediction_batch.py


CUDA_VISIBLE_DEVICES=1 \
DATASET_TO_USE="scirepeval_search_validation_evaluation" \
RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
MODEL_TO_USE="splade_based_custom_trained_combined_references_v6-2_position_penalty+length_penalty" \
PRECOMPUTED_REPRESENTATIONS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v6-2--scirepeval_search_validation_evaluation.jsonl" \
python run_keyphrase_prediction_batch.py



CUDA_VISIBLE_DEVICES=1 \
DATASET_TO_USE="scirepeval_search_validation_evaluation" \
RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_v6-2_position_penalty+length_penalty" \
PRECOMPUTED_REPRESENTATIONS_PATH="/scratch/lamdo/precompute_sparse_representations/custom_trained_combined_references_v6-2--scirepeval_search_validation_evaluation.jsonl" \
python run_keyphrase_prediction_batch.py





#--------

CUDA_VISIBLE_DEVICES=0 \
DATASET_TO_USE="trec_covid" \
RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_v6-2_position_penalty+length_penalty" \
python run_keyphrase_prediction_batch.py

CUDA_VISIBLE_DEVICES=1 \
DATASET_TO_USE="scidocs" \
RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_v6-2_position_penalty+length_penalty" \
python run_keyphrase_prediction_batch.py


CUDA_VISIBLE_DEVICES=2 \
DATASET_TO_USE="scifact" \
RESULTS_FOLDER="/scratch/lamdo/precompute_keyphrase_extraction/" \
MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_v6-2_position_penalty+length_penalty" \
python run_keyphrase_prediction_batch.py



#-----

CUDA_VISIBLE_DEVICES=2 \
DATASET_TO_USE="kp20k" \
RESULTS_FOLDER="/scratch/lamdo/keyphrase_generation_results/results_ongoing/" \
MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" \
python run_keyphrase_prediction_batch.py