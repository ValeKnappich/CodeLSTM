import tokenize
import io
from pathlib import Path
import json

from typing import Optional, List, Tuple
import logging

import torch
from tqdm import tqdm


def tokenizer(
        s: str, id: int, error_dict: dict, from_string: Optional[bool] = True
    ) -> List[tokenize.TokenInfo]:
    
    fp = open(s) if not from_string else io.StringIO(s)
    filter_types = [tokenize.ENCODING, tokenize.ENDMARKER, tokenize.ERRORTOKEN]
    tokens = []
    token_gen = tokenize.generate_tokens(fp.readline)
    while True:
        try:
            token = next(token_gen)
            if token.string and token.type not in filter_types:
                tokens.append(token)
        except tokenize.TokenError:
            error_dict["TokenError"].append(id)
            break
        except StopIteration:
            break
        except IndentationError:
            error_dict["IndentationError"].append(id)
            continue
    return tokens


def char_index_to_token_index(instance: dict, tokens: list) -> int:
    tokens = [token.string for token in tokens]
    wrong_code = instance["wrong_code"]
    target_char_i = instance["metadata"]["fix_location"]
    char_i = 0
    for token_i, token in enumerate(tokens):
        # Skip spaces and other non-token stuff
        char_i = wrong_code.find(token, char_i)
        if char_i <= target_char_i <= char_i + len(token) - 1:
            break
        char_i += len(token)
    return token_i


def print_error_stats(error_dict: dict, total_length: int) -> None:
    for error_name, ids in error_dict.items():
        if ids:
            logging.warning(
                f"{error_name} occurred {len(ids)} of {total_length} times at ID's "
                f"{', '.join(str(id) for id in ids[:10])}{'...' if len(ids) > 10 else ''}"
            )


def load_data(
        path: str, test_frac: float = 0.1, error_dict: dict = None
    ) -> Tuple[Tuple[List[tokenize.TokenInfo], List[tokenize.TokenInfo]], set]:

    vocab = set()
    data = []

    with path.open() as fp:
        data = json.load(fp)

    preprocessed_data = []

    # If this is the only file, print errors here, else in super method
    if not error_dict:
        print_errors = True
        error_dict = {"TokenError": [], "IndentationError": []}
    else:
        print_errors = False

    for instance in data:
        tokens = tokenizer(
            instance["wrong_code"],
            instance["metadata"]["id"],
            error_dict
        )
        instance_data = {
            "tokens": [token.string for token in tokens],
            "error_index":  char_index_to_token_index(instance, tokens)
        }
        preprocessed_data.append(instance_data)
        vocab.update((token.string for token in tokens))

    if print_errors:
        print_error_stats(error_dict, len(preprocessed_data))

    indices = torch.randperm(len(preprocessed_data))
    test_num = int(test_frac * len(indices))
    test = [preprocessed_data[i] for i in indices[:test_num]]
    train = [preprocessed_data[i] for i in indices[test_num:]]

    return (train, test), vocab


def load_multiple(
        path: str, test_frac: float = 0.1
    ) -> Tuple[Tuple[List[tokenize.TokenInfo], List[tokenize.TokenInfo]], set]:
    error_dict = {"TokenError": [], "IndentationError": []}
    train, test, vocab = [], [], set()
    files = list(Path(path).glob("*.json"))
    for file in tqdm(files, desc="Loading Data Files"):
        (train_, test_), vocab_ = load_data(file, test_frac=test_frac, error_dict=error_dict)
        train.extend(train_)
        test.extend(test_)
        vocab.update(vocab_)
    print_error_stats(error_dict, len(train) + len(test))
    return (train, test), vocab