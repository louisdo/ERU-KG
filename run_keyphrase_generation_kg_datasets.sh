
# models_types=( "embedrank_sent2vec" "embedrank_sentence_transformers_all-MiniLM-L6-v2 " "splade_based")
# models_types=("splade_based_custom_trained_scirepeval_search->unarxive_intro_relatedwork_1citationpersentence" "splade_based_custom_trained_scirepeval_search->unarxive_intro_relatedwork_1citationpersentence_position_penalty+length_penalty" "splade_based_custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search" "splade_based_custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search_position_penalty+length_penalty" "splade_based_custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search_v2" "splade_based_custom_trained_unarxive_intro_relatedwork_1citationpersentence+scirepeval_search_v2_position_penalty+length_penalty")
# models_types=("embedrank_sent2vec" "splade_based_custom_trained_combined_references_v3_position_penalty+length_penalty" "splade_based_custom_trained_combined_references_v4_position_penalty+length_penalty")
# models_types=("embedrank_sent2vec" "splade_based_custom_trained_combined_references_v3_position_penalty+length_penalty" "splade_based_custom_trained_combined_references_v4_position_penalty+length_penalty" "splade_based_custom_trained_scirepeval_search_v2_position_penalty+length_penalty")
# models_types=("multipartiterank")
# models_types=("splade_based_custom_trained_scirepeval_search_position_penalty" "splade_based_custom_trained_scirepeval_search_position_penalty+length_penalty")
# models_types=("embedrank_sent2vec" "splade_based_custom_trained_combined_references_v6_position_penalty+length_penalty" "splade_based_custom_trained_combined_references_v8_position_penalty+length_penalty")
# models_types=("keybart")
# models_types=("retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty" "embedrank_sent2vec" "uokg" "multipartiterank")
# models_types=( "multipartiterank" "embedrank_sent2vec" "retrieval_based_ukg_custom_trained_combined_references_v6_position_penalty+length_penalty" )
# models_types=("uokg-2" "uokg-3" "uokg-4" "uokg-5")
# models_types=("autokeygen-2" "autokeygen-3" "autokeygen-4" "autokeygen-5")
# models_types=("copyrnn-2" "copyrnn-3" "copyrnn-4" "copyrnn-5")

# models_types=("splade_based_custom_trained_combined_references_v11-2_position_penalty+length_penalty" "splade_based_custom_trained_combined_references_v11-3_position_penalty+length_penalty")

# models_types=("copyrnn-1")

# models_types=("autokeygen-1")

# models_types=("retrieval_based_ukg_custom_trained_combined_references_v6-2_position_penalty+length_penalty")
# models_types=(
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty"
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-2_position_penalty+length_penalty" 
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-3_position_penalty+length_penalty"
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-4_position_penalty+length_penalty"
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-5_position_penalty+length_penalty"
# )
# models_types=(
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-4_position_penalty+length_penalty"
#     "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-5_position_penalty+length_penalty" 
# )

# models_types=("embedrank_sent2vec")

models_types=(
    "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-1_position_penalty+length_penalty"
    "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-2_position_penalty+length_penalty" 
    "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-3_position_penalty+length_penalty"
    "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-4_position_penalty+length_penalty"
    "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-5_position_penalty+length_penalty"
)

datasets=("semeval" "inspec" "nus" "krapivin" "kp20k")
# datasets=("krapivin")
# datasets=("kp20k")
# datasets=("scirepeval_search_validation_evaluation")
# datasets=("scifact" "scidocs" "trec_covid" "nfcorpus" "doris_mae" "scifact_queries" "scidocs_queries" "trec_covid_queries" "nfcorpus_queries" "doris_mae_queries")
# datasets=("scifact_queries" "scidocs_queries")
# datasets=("trec_covid_queries")

# datasets=("semeval" "inspec" "nus" "krapivin")
# datasets=("trec_covid")

result_folder="/scratch/lamdo/keyphrase_generation_results/results_ongoing/"
# result_folder="/scratch/lamdo/precompute_keyphrase_extraction/"


# Nested loop
for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"
        # Add your desired actions here, e.g.:
        # - Call a function with the current combination
        # - Perform some operation on the combination
        # - Write the combination to a file

        CUDA_VISIBLE_DEVICES=1 DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction_batch.py
    done
done