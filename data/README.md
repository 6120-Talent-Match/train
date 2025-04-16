# Resume Dataset

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
dataset = load_dataset("C0ldSmi1e/resume-dataset")

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
dataset = load_dataset("C0ldSmi1e/resume-dataset")
train_data = dataset["train"]
eval_data = dataset["eval"]
```
