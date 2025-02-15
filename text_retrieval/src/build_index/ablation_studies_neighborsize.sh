datasets=(
    scidocs
    scifact
    trec_covid
    nfcorpus
    doris_mae
    acm_cr
)

neighborsizes=(
    10
    50
)

for dataset in "${datasets[@]}"; do
    for neighborsize in "${neighborsizes[@]}"; do
        OUTPUT_FOLDER="/scratch/lamdo/pyserini_experiments/${dataset}_collections/pyserini_formatted_collection_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-1_position_penalty+length_penalty_neighborsize_${neighborsize}" \
        KEYWORD_FOR_DOCUMENT_EXPANSION="/scratch/lamdo/precompute_keyphrase_extraction/${dataset}--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-1_position_penalty+length_penalty_neighborsize_${neighborsize}.json" \
        INDEX_FOLDER="/scratch/lamdo/pyserini_experiments/index/${dataset}_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v8-1_position_penalty+length_penalty_neighborsize_${neighborsize}" \
        EXPANSION_ONLY_PRESENT_KEYPHRASES=0 \
        NUMBER_OF_KEYWORDS_EACH_TYPE=10 \
        python ${dataset}.py

    done
done