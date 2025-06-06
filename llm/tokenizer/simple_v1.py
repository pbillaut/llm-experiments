from llm.tokenizer import DECODE_PATTERN, Vocab
from llm.tokenizer.naive import naive_tokenization


def generate_vocab(tokens: list[str]) -> Vocab:
    all_words = sorted(set(tokens))
    return {token: token_id for token_id, token in enumerate(all_words)}


class SimpleTokenizerV1:
    def __init__(self, vocab: Vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        pre_processed = naive_tokenization(text)
        try:
            token_ids = [self.str_to_int[s] for s in pre_processed]
        except KeyError as e:
            raise KeyError(f"Unknown token: {e}")
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in token_ids])
        text = DECODE_PATTERN.sub(r"\1", text)
        return text
