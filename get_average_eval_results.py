import pandas as pd

def compute_average_performance(data_path, num_datasets):
    # Load the dataset into a DataFrame
    df = pd.read_csv(data_path)
    # Group by model_name and count the number of unique datasets for each model
    dataset_counts = df.groupby("model_name")["dataset_name"].nunique()

    # Filter models that have evaluation results for all datasets
    valid_models = dataset_counts[dataset_counts == num_datasets].index

    # Filter the DataFrame to only include valid models
    valid_df = df[df["model_name"].isin(valid_models)].drop("dataset_name", axis=1)

    # Compute the average performance for each valid model
    average_performance = valid_df.groupby("model_name").mean().reset_index()

    return average_performance



results = compute_average_performance("view_full.csv", 5)


results.to_csv("view_average.csv", index = False)