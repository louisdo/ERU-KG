# CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" MODEL_TO_USE="copyrnn-1" python cpu_time_comparison.py
# 795.3976709599999

CUDA_VISIBLE_DEVICES=1 DATASET_TO_USE="semeval" RUN_INDEX=1 MODEL_TO_USE="promptrank" python cpu_time_comparison.py
# CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" RUN_INDEX=1 MODEL_TO_USE="promptrank" python cpu_time_comparison.py

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMBA_NUM_THREADS=1
# export VECLIB_MAXIMUM_THREADS=1
# export JAVA_OPTS="-Dorg.apache.lucene.search.ThreadPoolExecutor.numThreads=1"

# CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" RUN_INDEX=1 MODEL_TO_USE="promptrank" python cpu_time_comparison.py
# for i in $(seq 1 5);
# do
    # CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" RUN_INDEX=$i MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" python cpu_time_comparison.py
    # CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" RUN_INDEX=$i MODEL_TO_USE="copyrnn-1" python cpu_time_comparison.py
    # CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" RUN_INDEX=$i MODEL_TO_USE="autokeygen-1" python cpu_time_comparison.py
    # CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" RUN_INDEX=$i MODEL_TO_USE="uokg-1" python cpu_time_comparison.py
    # CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" RUN_INDEX=$i MODEL_TO_USE="tpg-1" python cpu_time_comparison.py
    # CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" RUN_INDEX=$i MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-1_position_penalty+length_penalty" python cpu_time_comparison.py
# done 