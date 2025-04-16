import os
import json
import pandas as pd
from huggingface_hub import HfApi, login, create_repo
from config import settings

# Set your Hugging Face API token
HF_TOKEN = settings.HUGGING_FACE_API_KEY
REPO_ID = "C0ldSmi1e/resume-dataset"  # Change to your desired repository name


def convert_json_to_csv(json_path, csv_train_path, csv_eval_path):
  """
  Convert JSON file to CSV format and split into train and eval sets.
  """
  print(f"Loading JSON data from {json_path}...")
  with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

  print(f"Converting JSON to DataFrame...")
  # For nested JSON with lists, we need to handle it specially
  processed_data = []
  for item in data:
    # Convert lists to string representation
    item_copy = item.copy()
    for key, value in item_copy.items():
      if isinstance(value, list):
        item_copy[key] = ', '.join(value)
    processed_data.append(item_copy)

  df = pd.DataFrame(processed_data)

  # Filter out rows with empty fields or None values
  initial_count = len(df)
  print(f"Initial dataset size: {initial_count}")

  # Replace empty strings with NaN for consistent filtering
  df = df.replace('', pd.NA)

  # Drop rows with any NA/None values
  df = df.dropna()

  # Check for any remaining empty strings (just in case)
  for col in df.columns:
    df = df[df[col].astype(str).str.strip().astype(bool)]

  filtered_count = len(df)
  removed_count = initial_count - filtered_count
  print(f"Removed {removed_count} rows with empty or None values")
  print(f"Filtered dataset size: {filtered_count}")

  # Split data: last 200 rows for eval, rest for train
  # Ensure we don't take too many for eval
  eval_size = min(200, int(filtered_count * 0.2))
  eval_df = df.iloc[-eval_size:].copy()
  train_df = df.iloc[:-eval_size].copy()

  print(f"Train set size: {len(train_df)}")
  print(f"Eval set size: {len(eval_df)}")

  # Create directory if it doesn't exist
  os.makedirs(os.path.dirname(csv_train_path), exist_ok=True)

  # Save as CSV
  train_df.to_csv(csv_train_path, index=False)
  eval_df.to_csv(csv_eval_path, index=False)

  print(f"Train data saved at {csv_train_path}")
  print(f"Eval data saved at {csv_eval_path}")
  print(f"CSV columns: {df.columns.tolist()}")

  return csv_train_path, csv_eval_path


def upload_to_huggingface():
  """
  Convert and upload train.json to Hugging Face as train and eval splits.
  """
  # Step 1: Convert JSON to CSV and split into train/eval
  json_path = "./data/train.json"
  csv_train_path = "./data/train.csv"
  csv_eval_path = "./data/eval.csv"
  csv_train_file, csv_eval_file = convert_json_to_csv(
      json_path, csv_train_path, csv_eval_path)

  # Step 2: Initialize the Hugging Face API
  api = HfApi()

  # Step 3: Login to Hugging Face
  print("Logging in to Hugging Face...")
  login(token=HF_TOKEN)

  # Step 4: Create repository if it doesn't exist
  try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False,  # Set to True if you want a private repository
        exist_ok=True
    )
    print(f"Repository {REPO_ID} is ready to use")
  except Exception as e:
    print(f"Error creating repository: {e}")
    raise

  # Step 5: Upload README file with dataset information
  readme_content = f"""# Resume Dataset

## Dataset Description

This dataset contains resume data for different job categories with skills, education, and experience information that can be used for resume classification or career prediction applications.

### Data Structure

This dataset is stored in CSV format with the following columns:
- `id`: Unique identifier for each resume
- `category`: Job category or field (e.g., HR, IT, Marketing)
- `skills`: Comma-separated list of skills mentioned in the resume
- `education`: Comma-separated list of education qualifications
- `experience`: Comma-separated list of job experiences

### Dataset Splits

- `train`: Main training dataset
- `eval`: Evaluation dataset (last 200 samples)

## Usage

You can load this dataset using the Hugging Face datasets library:

```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("{REPO_ID}")

# Access specific splits
train_data = dataset["train"]
eval_data = dataset["eval"]

# Check the columns
print(train_data.column_names)

# Access a sample entry
print(train_data[0])
```

## Example: Using with a Tokenizer

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
EOS_TOKEN = tokenizer.eos_token  # End of sequence token

# Load your dataset
dataset = load_dataset("{REPO_ID}")
train_data = dataset["train"]
eval_data = dataset["eval"]
```
"""

  with open("data/README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

  # Step 6: Upload CSV files
  print(f"Uploading CSV files to {REPO_ID}...")
  api.upload_file(
      path_or_fileobj=csv_train_file,
      path_in_repo="train.csv",
      repo_id=REPO_ID,
      repo_type="dataset"
  )

  api.upload_file(
      path_or_fileobj=csv_eval_file,
      path_in_repo="eval.csv",
      repo_id=REPO_ID,
      repo_type="dataset"
  )

  # Step 7: Upload README
  api.upload_file(
      path_or_fileobj="data/README.md",
      path_in_repo="README.md",
      repo_id=REPO_ID,
      repo_type="dataset"
  )

  # Step 8: Upload dataset card with metadata for the splits
  dataset_card = """
---
dataset_info:
  features:
    - name: id
      dtype: string
    - name: category
      dtype: string 
    - name: skills
      dtype: string
    - name: education
      dtype: string
    - name: experience
      dtype: string
  splits:
    - name: train
      num_examples: auto
    - name: eval
      num_examples: 200
  config_name: default
---
"""
  with open("data/dataset_card.yaml", "w", encoding="utf-8") as f:
    f.write(dataset_card)

  api.upload_file(
      path_or_fileobj="data/dataset_card.yaml",
      path_in_repo="dataset_card.yaml",
      repo_id=REPO_ID,
      repo_type="dataset"
  )

  print(f"Files successfully uploaded to {REPO_ID}")
  print("You can now load the dataset with:")
  print(f"""
from datasets import load_dataset

# Load the entire dataset with both splits
dataset = load_dataset("{REPO_ID}")

# Access specific splits
train_data = dataset["train"]
eval_data = dataset["eval"]

# Example with your code
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  
EOS_TOKEN = tokenizer.eos_token
print(train_data.column_names)
print(f"Train size: {{len(train_data)}}, Eval size: {{len(eval_data)}}")
""")


if __name__ == "__main__":
  upload_to_huggingface()
