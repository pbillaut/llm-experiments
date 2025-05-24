from pathlib import Path

from llm.tokenizer import naive_tokenization


def test_naive_tokenizer_the_verdict(resources_root: Path):
    path = resources_root / 'the-verdict.txt'
    raw_text = path.read_text()
    tokens = naive_tokenization(raw_text)
    assert len(tokens) == 4690
