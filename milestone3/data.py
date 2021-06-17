import tokenize
import io
from pathlib import Path
import json
import os
from typing import List, Tuple
import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


FIX_TYPES = ["insert", "modify", "delete"]


def tokenizer(
        s: str, id: int, error_dict: dict
    ) -> List[tokenize.TokenInfo]:
    
    filter_types = [tokenize.ENCODING, tokenize.ENDMARKER]
    token_gen = tokenize.generate_tokens(io.StringIO(s).readline)
    tokens = []
    # use while instead of for to be able to handle exceptions inside the generator
    while True:
        try:
            token = next(token_gen)
            if token.string and token.type not in filter_types:
                tokens.append(token)
        except tokenize.TokenError:
            error_dict["TokenError"].append(id)
            break
        except IndentationError:
            error_dict["IndentationError"].append(id)
            continue
        except StopIteration:
            break
    # Add empty token to enable model to classify that a token has to be appended
    # endmarker is not included when one of the errors above occur
    # -> remove all endmarkers and always add one in the end
    tokens.append(tokenize.TokenInfo(tokenize.ENDMARKER, "", len(s), len(s), ""))
    return tokens


def char_index_to_token_index(instance: dict, tokens: list) -> int:
    if isinstance(tokens[0], tokenize.TokenInfo):
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


def token_index_to_char_index(code: str, tokens: list, token_index: int, meta) -> int:
    char_i = 0
    for token in tokens[:token_index + 1]:
        char_i = code.find(token, char_i)
        char_i += len(token)
    char_i -= len(tokens[token_index]) # remove last to get start of error token
    return char_i


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
            "error_index":  char_index_to_token_index(instance, tokens),
            "fix_type": FIX_TYPES.index(instance["metadata"]["fix_type"]),
            "fix_token": instance["metadata"].get("fix_token"),
            "metadata": instance["metadata"],
            "wrong_code": instance["wrong_code"]
        }
        preprocessed_data.append(instance_data)
        vocab.update((token.string for token in tokens))
    
    # Assign token ids to tokens after vocab is final
    # Token order: PAD, UNK, vocab[0], ...
    vocab = ["PAD", "UNK"] + list(vocab)
    token_to_id = {token: id for id, token in enumerate(vocab)}
    id_to_token = {id: token for id, token in token_to_id.items()}
    for instance in preprocessed_data:
        instance["fix_token"] = id_to_token[instance["fix_token"]] if instance["fix_token"] else -1
        instance["tokens"] = [token_to_id[token] for token in instance["tokens"]]

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


def combine_batch(batch: List[dict]) -> Tuple[list, list, list, list]:
    # Merge tokens, labels etc. to form a batch
    input_ids    = [torch.tensor(instance["tokens"]) for instance in batch]
    input_ids    = pad_sequence(input_ids, batch_first=True)
    fix_location = torch.tensor([instance["error_index"] for instance in batch])
    fix_type     = torch.tensor([instance["fix_type"] for instance in batch])
    fix_token    = torch.tensor([instance["fix_token"] for instance in batch])
    meta_data    = [instance["metadata"] for instance in batch]
    wrong_code   = [instance["wrong_code"] for instance in batch]
    return input_ids, fix_location, fix_type, fix_token, meta_data, wrong_code


if __name__ == "__main__":
    # test data loading and conversion between char index and token index
    (train, _), vocab = load_multiple("dataset/", test_frac=0)
    errors = 0
    walrus = 0
    for i, instance in enumerate(train):
        char_i = instance["metadata"]["fix_location"]
        token_index = char_index_to_token_index(
            instance, instance["tokens"]
        )
        char_i_recon = token_index_to_char_index(
            instance["wrong_code"], instance["tokens"], token_index, instance["metadata"]
        )
        if char_i != char_i_recon:
            errors += 1
            if ":=" in instance["wrong_code"]:
                walrus += 1
    print(f"Found {errors} errors during reconstruction, {walrus} of those included the walrus operator")
