import pytest

from llm.tokenizer import (
    Vocab,
)
from llm.tokenizer.naive import naive_tokenization
from llm.tokenizer.simple_v1 import SimpleTokenizerV1, generate_vocab


@pytest.fixture(scope="module")
def text() -> str:
    return "The quick brown fox jumps over the lazy dog"


@pytest.fixture()
def vocab(text: str) -> Vocab:
    vocab_tokens = naive_tokenization(text)
    return generate_vocab(vocab_tokens)


def test_generate_vocabulary(text: str):
    vocab_tokens = naive_tokenization(text)
    vocab = generate_vocab(vocab_tokens)

    assert vocab == {
        "The": 0,
        "brown": 1,
        "dog": 2,
        "fox": 3,
        "jumps": 4,
        "lazy": 5,
        "over": 6,
        "quick": 7,
        "the": 8,
    }


def test_encode(vocab: Vocab, text: str):
    tokenizer = SimpleTokenizerV1(vocab)
    encoded = tokenizer.encode(text)

    assert encoded == [0, 7, 1, 3, 4, 6, 8, 5, 2]


def test_encode_fails_for_token_not_in_vocab(vocab: Vocab):
    tokenizer = SimpleTokenizerV1(vocab)
    unknown_token = "elephant"
    with pytest.raises(KeyError, match=unknown_token):
        tokenizer.encode(f"The quick brown {unknown_token} jumps over the lazy dog")


def test_decode(vocab: Vocab, text: str):
    tokenizer = SimpleTokenizerV1(vocab)
    decoded = tokenizer.decode([0, 7, 1, 3, 4, 6, 8, 5, 2])

    assert decoded == text
