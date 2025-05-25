import re
from typing import TypeAlias

Vocab: TypeAlias = dict[str, int]

DECODE_PATTERN = re.compile(r"""\s+([,.?!"'()])""")
