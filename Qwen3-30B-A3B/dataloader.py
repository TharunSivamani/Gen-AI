from litdata import StreamingDataset, StreamingDataLoader

# DataModule using LitData
class LitData(pl.LightningDataModule):
    def __init__(self, seq_length, batch_size, dataset_uri=None):
        super().__init__()
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.dataset_uri = dataset_uri or "hf://datasets/HuggingFaceTB/smollm-corpus/cosmopedia-v2"

    def train_dataloader(self):
        ds = StreamingDataset(
            self.dataset_uri,
            shuffle=True,
            drop_last=True,
        )

        def collate_fn(batch):
            # grab the "text" field
            texts = [ex["text"] for ex in batch]

            toks = tokenizer(
                texts,
                truncation=True,
                max_length=self.seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = toks.input_ids
            # ensure IDs are within vocab range
            input_ids = torch.clamp(input_ids, max=vocab_size - 1)

            return {
                "input_ids": input_ids,
                "labels": input_ids.clone()
            }

        return StreamingDataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )
