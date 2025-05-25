import pytest

from llm.tokenizer import (
    Vocab,
)
from llm.tokenizer.naive import naive_tokenization
from llm.tokenizer.simple_v2 import (
    SimpleTokenizerV2,
    TOKEN_END_OF_TEXT,
    TOKEN_UNKNOWN,
    generate_vocab,
)


@pytest.fixture(scope="module")
def text() -> str:
    return "The quick brown fox jumps over the lazy dog"


@pytest.fixture()
def vocab(text: str) -> Vocab:
    vocab_tokens = naive_tokenization(text)
    return generate_vocab(vocab_tokens)


def test_generate_vocabulary():
    text = "The quick brown fox jumps over the lazy dog"
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
        TOKEN_UNKNOWN: 9,
        TOKEN_END_OF_TEXT: 10,
    }


def test_encode(vocab: Vocab, text: str):
    tokenizer = SimpleTokenizerV2(vocab)
    encoded = tokenizer.encode(text)

    assert encoded == [0, 7, 1, 3, 4, 6, 8, 5, 2]


def test_encode_substitutes_unknown_token(vocab: Vocab):
    tokenizer = SimpleTokenizerV2(vocab)
    unknown_token = "elephant"
    encoded = tokenizer.encode(
        f"The quick brown {unknown_token} jumps over the lazy dog"
    )

    assert encoded == [0, 7, 1, 9, 4, 6, 8, 5, 2]


def test_decode(vocab: Vocab):
    tokenizer = SimpleTokenizerV2(vocab)
    decoded = tokenizer.decode([0, 7, 1, 9, 4, 6, 8, 5, 2])

    assert decoded == f"The quick brown {TOKEN_UNKNOWN} jumps over the lazy dog"
