from pathlib import Path

from llm.tokenizer.naive import naive_tokenization


def test_naive_tokenization(resource: Path):
    path = resource / "the-verdict.txt"
    raw_text = path.read_text()
    tokens = naive_tokenization(raw_text)
    assert len(tokens) == 4690
