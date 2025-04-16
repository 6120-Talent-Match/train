# 🧠 Resume Reader — LLM Fine-tuning Pipeline

This repository provides the full pipeline for fine-tuning a large language model to **extract structured information from resumes**, such as skills, education, and experience. It includes data preparation, prompt engineering, and training using the [Unsloth](https://github.com/unslothai/unsloth) framework.

You can access the trained model and dataset on Hugging Face:

- 📦 [Model: `C0ldSmi1e/resume-reader-best`](https://huggingface.co/C0ldSmi1e/resume-reader-best)
- 🗃️ [Dataset: `C0ldSmi1e/resume-dataset`](https://huggingface.co/datasets/C0ldSmi1e/resume-dataset)

---

## 📚 Dataset Overview

This resume dataset includes real-world resume content annotated with job categories. It can be used for tasks like classification, entity recognition, or instruction fine-tuning.

### Structure

Each entry contains:
- `ID`: Unique identifier
- `Resume_str`: Raw resume text
- `Resume_html`: HTML-formatted resume (optional)
- `Category`: Resume/job classification

### Load Example

```python
from datasets import load_dataset

dataset = load_dataset("C0ldSmi1e/resume-dataset", split="train")

print(dataset.column_names)
print(dataset[0])
```

### Use with Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
EOS_TOKEN = tokenizer.eos_token

def preprocess(examples):
    texts = [resume + EOS_TOKEN for resume in examples["Resume_str"]]
    return tokenizer(texts, truncation=True, max_length=512)

tokenized = dataset.map(preprocess, batched=True)
```

---

## 🏋️‍♀️ Training Pipeline

The model is fine-tuned using the [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B) via the Unsloth library.

Key features:
- Supports 4-bit quantization for low-memory training
- Prompt-based instruction tuning
- Max sequence length of 8096 tokens

You can review the full notebook-based training pipeline here:
📓 [`CS_6120_Final_Project_Pipeline.ipynb`](./CS_6120_Final_Project_Pipeline.ipynb)

---

## 💡 Inference Prompt Format

Example prompt used for instruction-tuning:

```
You are an experienced HR and now you will review a resume then extract key information from it.

# Input
Here is the resume text:
[RESUME TEXT HERE]

### Response
<think>
```

The model will output JSON with the following structure:

```json
{
  "skills": [...],
  "education": [...],
  "experience": [...]
}
```

---

## 🚀 Quick Links

- 🤗 Model: [resume-reader-best](https://huggingface.co/C0ldSmi1e/resume-reader-best)
- 🤗 Dataset: [resume-dataset](https://huggingface.co/datasets/C0ldSmi1e/resume-dataset)
- 🧪 Notebook: [`CS_6120_Final_Project_Pipeline.ipynb`](./CS_6120_Final_Project_Pipeline.ipynb)

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

Made with ❤️ for CS6120 Final Project