# huge props to skyzip and jbm on https://stackoverflow.com/questions/69531811/using-hugginface-transformers-and-tokenizers-with-a-fixed-vocabulary
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

from transformers import PreTrainedTokenizer

class SPTTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab: Union[Dict[str, int], str], max_len: int = None):
        if isinstance(vocab, str):
            vocab_path = Path(vocab)
            with open(vocab_path, 'r') as f:
                self._token_ids = json.load(f)
        else:
            self._token_ids = vocab
            
        self._id_tokens: Dict[int, str] = {value: key for key, value in self._token_ids.items()}
        super().__init__(max_len=max_len)

        # Initialize special tokens for RoBERTa
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.mask_token = '<mask>'
        self.unk_token_id = self._token_ids.get(self.unk_token, 0)
        self.pad_token_id = self._token_ids.get(self.pad_token, 1)
        self.bos_token_id = self._token_ids.get(self.bos_token, 2)
        self.eos_token_id = self._token_ids.get(self.eos_token, 3)
        self.mask_token_id = self._token_ids.get(self.mask_token, 4)

    def _tokenize(self, text: str, **kwargs):
        return text.split(' ')

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_ids[token] if token in self._token_ids else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_tokens[index] if index in self._id_tokens else self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        return self._token_ids.copy()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ''
        vocab_path = Path(save_directory, filename_prefix + 'vocab.json')
        with open(vocab_path, 'w') as f:
            json.dump(self._token_ids, f)
        return (str(vocab_path),)

    @property
    def vocab_size(self) -> int:
        return len(self._token_ids)

# char_to_id = {}
# char_to_id["<bos>"] = 0

# for i in range(1,101):
#     char_to_id[str(i)] = i
# char_to_id["+"] = i + 1
# char_to_id["="] = i + 2
# char_to_id["<pad>"] = i + 3
# char_to_id["<unk>"] = i + 4
# char_to_id["<mask>"] = i + 5
# char_to_id["<eos>"] = i + 6

# with open('tokenizer/vocab.json', 'w') as f:
#     json.dump(char_to_id, f)

# print(char_to_id)
# sum_string_ex = "<bos>18+19=37<eos>"
# model_max_len = 8

# # Optionally specify the path to a vocab file
# vocab_path = 'tokenizer/vocab.json'

# # You can either pass the custom vocab dictionary or the path to the vocab file
# tokenizer = SPTTokenizer(vocab_path, max_len=model_max_len)

# res = tokenizer(
#     [
#         "<bos> 18 + 19 = 37 <eos>",
#         "<bos> 2 + 43 = 45 <eos>",
#     ],
#     padding=True,
#     truncation=True,
# )
# print(res)