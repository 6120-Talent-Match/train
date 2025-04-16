# Resume Dataset

## Dataset Description

This dataset contains resume data from job seekers with their professional experience, skills, and other details.

### Data Structure

Each entry in the dataset contains the following fields:
- `ID`: Unique identifier for the resume
- `Resume_str`: The full resume text content
- `Resume_html`: HTML formatted version of the resume (if available)
- `Category`: Job category or classification

## Usage

You can load this dataset using the Hugging Face datasets library:

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("C0ldSmi1e/resume-dataset", split="train")

# View the columns
print(dataset.column_names)

# Access a sample
print(dataset[0])

# Load a subset of the dataset
subset = load_dataset("C0ldSmi1e/resume-dataset", split="train[0:200]")
print(f"Subset size: {len(subset)}")
```

## Example: Using with a Tokenizer

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
EOS_TOKEN = tokenizer.eos_token  # End of sequence token

# Load a subset of data
dataset = load_dataset("C0ldSmi1e/resume-dataset", split="train[0:200]")

# Example processing and tokenization
def preprocess(examples):
    texts = [resume + EOS_TOKEN for resume in examples["Resume_str"]]
    return tokenizer(texts, truncation=True, max_length=512)

tokenized_dataset = dataset.map(preprocess, batched=True)
```

## License

[Insert appropriate license information here]

## Citation

If you use this dataset in your research, please cite:

```
@dataset{resume_dataset,
  author    = {Your Name},
  title     = {Resume Dataset},
  year      = {2023},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/C0ldSmi1e/resume-dataset}
}
``` 