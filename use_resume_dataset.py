from datasets import load_dataset
from transformers import AutoTokenizer

# Replace with your username and dataset name
USERNAME = "C0ldSmi1e"
DATASET_NAME = f"{USERNAME}/resume-dataset"

# Load the dataset
dataset = load_dataset(DATASET_NAME, split="train")

# Print dataset information
print(f"Dataset size: {len(dataset)}")
print(f"Dataset columns: {dataset.column_names}")
print(f"Sample entry: {dataset[0]}")

# Example: Load a tokenizer and tokenize a resume
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
EOS_TOKEN = tokenizer.eos_token  # End of sequence token


def tokenize_function(examples):
  # Add EOS token to the end of each resume
  texts = [text if text else "" + EOS_TOKEN for text in examples["text"]]
  return tokenizer(texts, truncation=True, padding="max_length", max_length=512)


# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print(f"Tokenized dataset features: {tokenized_dataset.column_names}")
print(
    f"First tokenized entry input_ids shape: {len(tokenized_dataset[0]['input_ids'])}")

# Example of filtering the dataset
filtered_dataset = dataset.filter(
    lambda example: len(example["text"]) > 1000)
print(f"Filtered dataset size: {len(filtered_dataset)}")

# Example of creating a small subset
small_dataset = dataset.select(range(200))
print(f"Small dataset size: {len(small_dataset)}")
