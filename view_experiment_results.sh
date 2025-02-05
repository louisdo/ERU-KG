models_to_include=(
    # "tpg-1"
    # "tpg-2"
    # "tpg-3"
    # "autokeygen-1"
    # "autokeygen-2"
    # "autokeygen-3"
    # "uokg-1"
    # "uokg-2"
    # "uokg-3"
    # "copyrnn-1"
    # "copyrnn-2"
    # "copyrnn-3"
    # "textrank"
    # "multipartiterank"
    # "promptrank"
    # "embedrank_sent2vec"
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty"
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-2_position_penalty+length_penalty"
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-3_position_penalty+length_penalty"
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-1_position_penalty+length_penalty"
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-2_position_penalty+length_penalty"
    # "retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-3_position_penalty+length_penalty"
    retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_10
    retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_50
)

join_by() {
  local separator="$1"
  shift
  local first="$1"
  shift
  printf "%s" "$first" "${@/#/$separator}"
}

models_to_include_joined=$(join_by , "${models_to_include[@]}")

echo "$models_to_include_joined"

RESULTS_FOLDER="/scratch/lamdo/keyphrase_generation_results/results_ongoing/" \
DATASETS_TO_INCLUDE="semeval,inspec,nus,krapivin,kp20k" MODELS_TO_INCLUDE=$models_to_include_joined python view_experiment_results.py

# "semeval,inspec,nus,krapivin,kp20k"