embeddings_folder="/scratch/lamdo/arxiv_classification/embeddings"
dataset_name="arxiv_classification"
output_file="experiments/arxiv.jsonl"

# no expansion
EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--arxiv_classification.json" \
EXPERIMENT_NAME="no_expansion" \
OUTPUT_FILE=$output_file \
python custom_evaluation_arxiv.py

for i in $(seq 1 3);
do
    EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--arxiv_classification_keyphrase_expansion_retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty.json" \
    EXPERIMENT_NAME="eru-kg-${i}" \
    OUTPUT_FILE=$output_file \
    python custom_evaluation_arxiv.py

    EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--arxiv_classification_keyphrase_expansion_copyrnn-${i}.json" \
    EXPERIMENT_NAME="copyrnn-${i}" \
    OUTPUT_FILE=$output_file \
    python custom_evaluation_arxiv.py

    EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--arxiv_classification_keyphrase_expansion_autokeygen-${i}.json" \
    EXPERIMENT_NAME="autokeygen-${i}" \
    OUTPUT_FILE=$output_file \
    python custom_evaluation_arxiv.py

    EMBEDDING_FILE="${embeddings_folder}/embeddings--specter2_base--arxiv_classification_keyphrase_expansion_uokg-${i}.json" \
    EXPERIMENT_NAME="uokg-${i}" \
    OUTPUT_FILE=$output_file \
    python custom_evaluation_arxiv.py
done