UNARXIVE_FOLDER_NAME="/scratch/lamdo/unArxive"
ARXIVID2METADATA_FILE="/scratch/lamdo/arxiv_dataset/arxivid2metadata.json"

CITATION_CONTEXT_INTERMEDIATE_OUTPUT_FILE=/scratch/lamdo/unArxive/triplets_full_paper_1citationpersentence_hardneg.json
CITATION_CONTEXT_TSV_OUTPUT_FILE=/scratch/lamdo/unArxive/keyphrase_informativeness_unArxiv/triplets_full_paper_1citationpersentence_hardneg/raw.tsv

TITLES_INTERMEDIATE_OUTPUT_FILE=/scratch/lamdo/unArxive/triplets_title_abstract_hardneg.json
TITLES_TSV_OUTPUT_FILE=/scratch/lamdo/unArxive/keyphrase_informativeness_unArxiv/triplets_title_abstract_hardneg/raw.tsv

QUERIES_TSV_OUTPUT_FILE=/home/lamdo/keyphrase_informativeness_test/splade/data/keyphrase_informative_scirepeval/triplets/raw.tsv

COMBINED_DATASET_TSV_OUTPUT_FILE=/scratch/lamdo/unArxive/keyphrase_informativeness_combined_references/triplets_hardneg/raw.tsv

# process citation context data
python citation_contexts.py \
--input_folder $UNARXIVE_FOLDER_NAME \ 
--arxive_metadata_file $ARXIVID2METADATA_FILE \
--output_file $CITATION_CONTEXT_INTERMEDIATE_OUTPUT_FILE


python convert_to_tsv.py \
--input_file $CITATION_CONTEXT_INTERMEDIATE_OUTPUT_FILE \
--output_file $CITATION_CONTEXT_TSV_OUTPUT_FILE


# process title data
python titles.py \
--input_folder $UNARXIVE_FOLDER_NAME \ 
--arxive_metadata_file $ARXIVID2METADATA_FILE \
--output_file $TITLES_INTERMEDIATE_OUTPUT_FILE

python convert_to_tsv.py \
--input_file $TITLES_INTERMEDIATE_OUTPUT_FILE \
--output_file $TITLES_TSV_OUTPUT_FILE


# process queries data

python queries.py \
--output_file $QUERIES_TSV_OUTPUT_FILE


# combine the data
python combine_datasets.py \
--queries_data_file $QUERIES_TSV_OUTPUT_FILE \
--citation_contexts_data_file $CITATION_CONTEXT_TSV_OUTPUT_FILE \
--titles_data_file $TITLES_TSV_OUTPUT_FILE \
--output_file $COMBINED_DATASET_TSV_OUTPUT_FILE