import os, requests, zipfile, sys


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



def download_and_unzip(url: str, extract_to: str):
    # Ensure the target directory exists
    os.makedirs(extract_to, exist_ok=True)

    # Download the file with progress
    local_zip_path = os.path.join(extract_to, 'downloaded_file.zip')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = r.headers.get('content-length')

        if total_length is None:  # No content length header
            with open(local_zip_path, 'wb') as f:
                f.write(r.content)
            print("Downloaded without progress (no content length available)")
        else:
            total_length = int(total_length)
            downloaded = 0
            with open(local_zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        done = int(50 * downloaded / total_length)
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded * 100 / total_length:.2f}%")
                        sys.stdout.flush()
            print()  # Newline after progress bar

    # Unzip the file
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Optionally remove the zip file after extraction
    os.remove(local_zip_path)

    print(f"File downloaded from {url} and extracted to {extract_to}")