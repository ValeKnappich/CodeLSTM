#!/usr/bin/python3

import argparse
import tokenize
import io
import json
from typing import Optional, List

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="File containing Python code to be parsed.", required=True)
parser.add_argument(
    '--destination', help="Path to output json file of extracted tokens.", required=True)


def to_token_list(s: str):
    with open(s, "rb") as fp:
        filter_types = [tokenize.ENCODING, tokenize.ENDMARKER, tokenize.ERRORTOKEN]
        return [
            t for t in tokenize.tokenize(fp.readline)
            if t.string and t.type not in filter_types
        ]


def write_tokens(tokens: List[str], destination: str):
    json.dump(
        list(t.string for t in tokens),
        open(destination, "w"),
        indent=2
    )

if __name__ == "__main__":
    args = parser.parse_args()

    # extract tokens for the code.
    tokens = to_token_list(args.source)

    # write extracted tokens to file.
    write_tokens(tokens, args.destination)
