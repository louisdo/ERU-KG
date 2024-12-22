# Classification-based Evaluation


## Set up evaluation framework
Cloning SciRepEval repo

```
cd ERU-KG
git clone git@github.com:allenai/scirepeval.git
```

Copy 4 files into `scirepeval` folder
```
cd ERU-KG
cp classification/custom_evaluation_fos.py scirepeval/
cp classification/custom_evaluation_fos.sh scirepeval/
cp classification/custom_generate_embeddings.py scirepeval/
cp classification/custom_generate_embeddings.py scirepeval/
```



## Data preparation
Run data preprocessing. One would need to change the variable `dataset_name` in `prepare_data_for_evaluation.sh`. The two dataset that we are using in this evaluation are `scirepeval_fos_test` and `scirepeval_mesh_descriptors_test`

```
cd ERU-KG/classification
./prepare_data_for_evaluation.sh
```

## Run evaluation
First generate embeddings

```
cd ERU-KG/scirepeval
./custom_generate_embeddings.sh
```

After that we can run evaluation (fit Linear SVC on generated embeddings). 

NOTE: The training will take a while to finish

```
cd ERU-KG/scirepeval
./custom_evaluation_fos.sh
```