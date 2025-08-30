from datasets import load_dataset
from torch.utils.data import DataLoader

def dataset_loader(dataset_name, subset, split, tokenizer, batch_size, seq_length):
    def encode(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=seq_length,
            return_tensors="pt",
            padding="max_length",
        )
        input_ids = tokens['input_ids'].squeeze(0)
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()
        }

    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        name="cosmopedia-v2",
        split="train",
        streaming=True
    )
    dataset = dataset.map(encode, remove_columns=["text"])
    dataset = dataset.with_format("torch")

    return DataLoader(dataset, batch_size=batch_size)
