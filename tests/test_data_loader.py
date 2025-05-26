from pathlib import Path

import torch

from llm.dataset import create_data_loader


def test_data_loader(resource: Path):
    path = resource / "the-verdict.txt"
    text = path.read_text(encoding="utf-8")

    data_loader = create_data_loader(
        text=text,
        batch_size=1,
        max_length=4,
        stride=1,
        shuffle=False,
    )

    data_iter = iter(data_loader)
    inputs, targets = next(data_iter)

    assert torch.equal(inputs, torch.tensor([[40, 367, 2885, 1464]]))
    assert torch.equal(targets, torch.tensor([[367, 2885, 1464, 1807]]))
