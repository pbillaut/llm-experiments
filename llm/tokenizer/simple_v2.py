from collections import defaultdict

from llm.tokenizer import DECODE_PATTERN, Vocab
from llm.tokenizer.naive import naive_tokenization


def generate_vocab(tokens: list[str]) -> Vocab:
    all_words = sorted(set(tokens))
    all_words.extend([TOKEN_UNKNOWN, TOKEN_END_OF_TEXT])
    return {token: token_id for token_id, token in enumerate(all_words)}


class SimpleTokenizerV2:
    def __init__(self, vocab: Vocab):
        self.str_to_int = defaultdict(lambda: vocab[TOKEN_UNKNOWN], vocab)
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        pre_processed = naive_tokenization(text)
        token_ids = [self.str_to_int[s] for s in pre_processed]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in token_ids])
        text = DECODE_PATTERN.sub(r"\1", text)
        return text


TOKEN_UNKNOWN = "<|unk|>"
TOKEN_END_OF_TEXT = "<|endoftext|>"
