import os


def maybe_create_folder(folder_path):
    """
    Creates the folder at folder_path if it does not exist.

    Args:
        folder_path (str): The path to the folder to create.
    """
    try:
        os.makedirs(folder_path, exist_ok=True)
        # print(f"Folder ensured at: {folder_path}")
    except Exception as e:
        print(f"Error creating folder '{folder_path}': {e}")