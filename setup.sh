pip install torchdata
python -c "from datasets import load_from_disk; \
data_paths='data/sweperf_training_0812'; \
dataset = load_from_disk(data_paths); \
parquet_path = os.path.join(data_paths,"data.parquet"); \
dataset.to_parquet(parquet_path)"
python -c "from datasets import load_from_disk; \
data_paths='data/sweperf_testing_0812'; \
dataset = load_from_disk(data_paths); \
parquet_path = os.path.join(data_paths,"data.parquet"); \
dataset.to_parquet(parquet_path)"