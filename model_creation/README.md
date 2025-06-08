# Model creation

Here we give instructions on how to build a keyphrase generation model from scratch. We acknowledge that the pipeline at the moment is not very convenient, and we will spend time improving it.

## SPLADE training
First we need to change directory

```bash
cd splade
```
The configuration for training is in the folder `splade/conf`. The files that we need to train our models are
+ `config_keyphrase_informativeness_combined_references`
+ `train/config/splade_keyphrase_informativeness_normal.yaml`
+ `train/data/keyphrase_informativeness_combined_references.yaml`
+ `train/model/splade_pretrain_from_distilbert` if you want to train using DistilBERT as your base model (similar to ERU-KG-base) or `train/model/splade_pretrain_from_bert_l6-512.yaml` if you want to use `google/bert_uncased_L-6_H-512_A-8` as your base model (similar to ERU-KG-small)

Please make appropriate adjustments for these files to train SPLADE with your custom dataset. More information on training SPLADE can be found in the [original repository](https://github.com/naver/splade)



After your model has done training, add it to the [configuration file](../erukg/erukg_helper/config.py). Add your model to `MODEL_NAME_2_MODEL_INFO`.


## Index creation

We need to create index for the phraseness module. To do this, first we need to create document vectors using the SPLADE model that we trained above. We do this by running the following script. We note that the following scripts will build index using the SciRepEval dataset (as described in our paper). We will provide instructions for using a custom dataset after.

```bash
CUDA_VISIBLE_DEVICES=2 \
DATASET_TO_USE="scirepeval_search_validation_evaluation" \
INFORMATIVENESS_MODEL_NAME={your_model_name} \ # Please replace it with your model name here
OUTPUT_FOLDER="/YOUR/OUTPUT/FOLDER" \ 
python precompute_splade_representations.py
```

The results will be stored in `/YOUR/OUTPUT/FOLDER/{your_model_name}--scirepeval_search_validation_evaluation.jsonl`

Next, we need to extract noun phrases from the documents within the corpus. This is achieved by running the following script

```bash
models_types=(
    nounphrase_extraction_1_5
)
datasets=(
    scirepeval_search_validation_evaluation
)
result_folder="/YOUR/OUTPUT/FOLDER"

for dataset in "${datasets[@]}"; do
    for model_type in "${models_types[@]}"; do
        echo "Config: $dataset - $model_type - $top_k"

        DATASET_TO_USE=$dataset RESULTS_FOLDER=$result_folder MODEL_TO_USE=$model_type python run_keyphrase_prediction.py
    done
done
```

The extracted noun phrases for the documents within the corpus will be stored in `/YOUR/OUTPUT/FOLDER/scirepeval_search_validation_evaluation--nounphrase_extraction_1_5.json`


Finally, run the following script

```bash
cd text_retrieval/src/build_index


INPUT_FILE="/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search_validation_evaluation.jsonl"  \
OUTPUT_FOLDER="/YOUR/INTERMEDIATE/OUTPUT/FOLDER" \
INDEX_FOLDER="/YOUR/INDEX/FOLDER" \
KEYWORD_FOR_DOCUMENT_EXPANSION="/YOUR/OUTPUT/FOLDER/scirepeval_search_validation_evaluation--nounphrase_extraction_1_5.json" \
DOCUMENT_VECTORS_PATH="/YOUR/OUTPUT/FOLDER/{your_model_name}--scirepeval_search_validation_evaluation.jsonl" \
python scirepeval_search_v2.py
```

The built index will be located at `/YOUR/INDEX/FOLDER` (you must name this folder using the same name as your SPLADE model). Finally, move this folder to `~/.cache/erukg_cache/ret_indexes`. After this you can [use your model](../introduction.ipynb)



### Using your own dataset
To build index with your own dataset. Please modify the [process_dataset function](../erukg/process_dataset.py) to include your dataset.