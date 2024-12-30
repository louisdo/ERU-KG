# CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" MODEL_TO_USE="copyrnn-1" python cpu_time_comparison.py
# 795.3976709599999

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export JAVA_OPTS="-Dorg.apache.lucene.search.ThreadPoolExecutor.numThreads=1"


for i in $(seq 1 5);
do
    CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-${i}_position_penalty+length_penalty" python cpu_time_comparison.py
    CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" MODEL_TO_USE="copyrnn-${i}" python cpu_time_comparison.py
    CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" MODEL_TO_USE="autokeygen-${i}" python cpu_time_comparison.py
    CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" MODEL_TO_USE="uokg-${i}" python cpu_time_comparison.py
done 