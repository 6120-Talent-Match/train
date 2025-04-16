import os
import pandas as pd
from huggingface_hub import HfApi, login, create_repo
from config import settings

# Function to convert data to CSV format


def convert_to_csv(data, output_path):
  """
  Convert data to CSV format and save it to the specified path.

  Args:
      data: The data to convert (can be a DataFrame, dict, list, etc.)
      output_path: Path where the CSV should be saved

  Returns:
      str: Path to the saved CSV file
  """
  # If data is already a DataFrame, use it directly
  if isinstance(data, pd.DataFrame):
    df = data
  # If data is a dict or list, convert to DataFrame
  elif isinstance(data, (dict, list)):
    df = pd.DataFrame(data)
  else:
    raise TypeError(
        "Unsupported data type. Please provide a DataFrame, dict, or list.")

  # Create directory if it doesn't exist
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  # Save as CSV
  df.to_csv(output_path, index=False)
  print(f"Data successfully converted to CSV and saved at {output_path}")

  return output_path

# Function to upload a single file to Hugging Face Hub


def upload_file_to_hf(file_path, repo_id, path_in_repo=None, repo_type="dataset", token=None, private=False):
  """
  Upload a single file to Hugging Face Hub.

  Args:
      file_path: Path to the local file to upload
      repo_id: ID of the repository (e.g., "username/dataset-name")
      path_in_repo: Path within the repository where the file should be placed
                   If None, the file will be placed at the root with the same name
      repo_type: Type of repository ("dataset", "model", or "space")
      token: Hugging Face API token. If None, will use the token from login()
      private: Whether to create a private repository (default: False)
  """
  # Initialize the Hugging Face API
  api = HfApi()

  # Login if token is provided
  if token:
    login(token=token)

  # If path_in_repo is not specified, use the filename only
  if path_in_repo is None:
    path_in_repo = os.path.basename(file_path)

  # Check if repository exists, create it if not
  try:
    # Try to create the repository (if it doesn't exist)
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True  # Won't raise error if repo already exists
    )
    print(f"Repository {repo_id} is ready to use")

    # Upload the file
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type
    )
    print(f"File successfully uploaded to {repo_id}/{path_in_repo}")

  except Exception as e:
    print(f"Error uploading file: {e}")
    raise

# Function to upload a folder to Hugging Face Hub


def upload_folder_to_hf(folder_path, repo_id, path_in_repo=None, repo_type="dataset", token=None, private=False):
  """
  Upload an entire folder to Hugging Face Hub.

  Args:
      folder_path: Path to the local folder to upload
      repo_id: ID of the repository (e.g., "username/dataset-name")
      path_in_repo: Path within the repository where the folder should be placed
                   If None, the folder content will be placed at the root
      repo_type: Type of repository ("dataset", "model", or "space")
      token: Hugging Face API token. If None, will use the token from login()
      private: Whether to create a private repository (default: False)
  """
  # Initialize the Hugging Face API
  api = HfApi()

  # Login if token is provided
  if token:
    login(token=token)

  # Check if repository exists, create it if not
  try:
    # Try to create the repository (if it doesn't exist)
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True  # Won't raise error if repo already exists
    )
    print(f"Repository {repo_id} is ready to use")

    # Upload the folder
    api.upload_folder(
        folder_path=folder_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type
    )
    print(f"Folder successfully uploaded to {repo_id}")

  except Exception as e:
    print(f"Error uploading folder: {e}")
    raise


# Example usage
if __name__ == "__main__":
  # Set your Hugging Face API token
  HF_TOKEN = settings.HUGGING_FACE_API_KEY

  # Example 1: Convert data and upload a single file
  data = {
      "column1": [1, 2, 3, 4, 5],
      "column2": ["A", "B", "C", "D", "E"]
  }
  csv_path = convert_to_csv(data, "data/example.csv")
  upload_file_to_hf(
      file_path=csv_path,
      repo_id="C0ldSmi1e/resume-dataset",  # Updated to match your repository ID
      path_in_repo="data/example.csv",
      token=HF_TOKEN,
      private=True  # Set to True if you want a private repository
  )

  # Example 2: Upload a folder containing multiple CSV files
  # Assuming you have multiple CSV files in the "data" folder
  upload_folder_to_hf(
      folder_path="data",
      repo_id="C0ldSmi1e/resume-dataset",  # Updated to match your repository ID
      token=HF_TOKEN,
      private=True  # Set to True if you want a private repository
  )
