import re

NAIVE_TOKENS = re.compile(r'''([_,.;:?!"'()]|--|\s)''')


def naive_tokenization(text: str) -> list[str]:
    tokens = NAIVE_TOKENS.split(text)
    return [text.strip() for text in tokens if text.strip()]
