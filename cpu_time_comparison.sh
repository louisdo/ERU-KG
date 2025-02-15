# CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="combined_kg" MODEL_TO_USE="copyrnn-1" python cpu_time_comparison.py
# 795.3976709599999

CUDA_VISIBLE_DEVICES=1 DATASET_TO_USE="semeval" RUN_INDEX=1 MODEL_TO_USE="promptrank" python cpu_time_comparison.py
CUDA_VISIBLE_DEVICES=-1 DATASET_TO_USE="semeval" RUN_INDEX=1 MODEL_TO_USE="promptrank" python cpu_time_comparison.py


# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMBA_NUM_THREADS=1
# export VECLIB_MAXIMUM_THREADS=1
# export JAVA_OPTS="-Dorg.apache.lucene.search.ThreadPoolExecutor.numThreads=1"
export CUDA_VISIBLE_DEVICES=-1

for i in $(seq 1 5);
do
    # DATASET_TO_USE="semeval" RUN_INDEX=$i MODEL_TO_USE="nounphrase_extraction_1_5" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="embedrank_sentence_transformers_all-MiniLM-L6-v2" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_10" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-1_position_penalty+length_penalty" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-1_position_penalty+length_penalty_neighborsize_10" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="copyrnn-1" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="autokeygen-1" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="uokg-1" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="tpg-1" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="embedrank_sent2vec" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="embedrank_sentence_transformers_all-MiniLM-L6-v2" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="embedrank_sentence_transformers_all-MiniLM-L12-v2" python cpu_time_comparison.py
    DATASET_TO_USE="semeval" RUN_INDEX=$i MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_alpha_1" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="retrieval_based_ukg_custom_trained_combined_references_nounphrase_v7-1_position_penalty+length_penalty_alpha_1" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="multipartiterank" python cpu_time_comparison.py
    # DATASET_TO_USE="combined_kg" RUN_INDEX=$i MODEL_TO_USE="textrank" python cpu_time_comparison.py
done 