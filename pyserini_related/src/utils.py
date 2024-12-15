

def convert_dataset_to_pyserini_format(dataset, dataset_name):
    if dataset_name == "nq320k":
        # dataset is expected to be a list
        assert isinstance(dataset, list)

        documents = []
        for i, line in enumerate(dataset):
            documents.append({"id": i, "contents": line})

        return documents