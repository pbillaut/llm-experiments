import tiktoken
import torch
from tiktoken import Encoding
from torch.utils.data import DataLoader, Dataset


def create_data_loader(
    text: str,
    max_length: int,
    stride: int,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GptDatasetV1(
        text=text,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


class GptDatasetV1(Dataset):
    def __init__(self, text: str, tokenizer: Encoding, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.input_ids[index], self.target_ids[index]
