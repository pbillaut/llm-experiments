from pathlib import Path

import pytest

from llm.tokenizer import SimpleTokenizerV1, Vocab, generate_vocab, naive_tokenization


def test_naive_tokenizer_the_verdict(resource: Path):
    path = resource / "the-verdict.txt"
    raw_text = path.read_text()
    tokens = naive_tokenization(raw_text)
    assert len(tokens) == 4690


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
    }


class TestSimpleTokenizerV1:
    TEXT = "The quick brown fox jumps over the lazy dog"

    @pytest.fixture()
    def vocab(self) -> Vocab:
        vocab_tokens = naive_tokenization(self.TEXT)
        return generate_vocab(vocab_tokens)

    def test_encode(self, vocab: Vocab):
        tokenizer = SimpleTokenizerV1(vocab)
        encoded = tokenizer.encode(self.TEXT)

        assert encoded == [0, 7, 1, 3, 4, 6, 8, 5, 2]

    def test_encode_fails_for_token_not_in_vocab(self, vocab: Vocab):
        tokenizer = SimpleTokenizerV1(vocab)
        unknown_token = "elephant"
        with pytest.raises(KeyError, match=unknown_token):
            tokenizer.encode(f"The quick brown {unknown_token} jumps over the lazy dog")

    def test_decode(self, vocab: Vocab):
        tokenizer = SimpleTokenizerV1(vocab)
        decoded = tokenizer.decode([0, 7, 1, 3, 4, 6, 8, 5, 2])

        assert decoded == self.TEXT
