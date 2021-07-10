import argparse
from pathlib import Path
import json
from typing import List

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data import combine_batch, load_data, load_multiple, token_index_to_char_index, FIX_TYPES, is_correct
from Train import CodeLSTM      # 'useless' import is needed for torch to load model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', help="Path of trained model.", default="model.pth")
parser.add_argument(
    '--source', help="Folder path of all test files.", default="../dataset/test")
parser.add_argument(
    '--destination', help="Path to output json file of extracted predictions.", default="predictions.json")


def predict(model: torch.nn.Module, test_files: str):
    # Enable loading multiple files
    if test_files.is_file():
        (train_ds, _), _ = load_data(test_files, test_frac=0, vocab=model.vocab)
    elif test_files.is_dir():
        (train_ds, _), _ = load_multiple(test_files, test_frac=0, vocab=model.vocab)
    else:
        raise FileNotFoundError(f"File or directory {test_files.absolute()} was not found")

    dl = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=combine_batch)
    metas, preds = [], []
    for input_ids, tokens, fix_location, fix_type, fix_token, meta_data, wrong_code in tqdm(dl, desc="Predicting"):
        input_ids      = input_ids.to(DEVICE)
        fix_location   = fix_location.to(DEVICE)
        fix_type       = fix_type.to(DEVICE)
        fix_token_mask = fix_token != -1        # filter out the instances where there is no token to predict (delete type)
        fix_token      = fix_token[fix_token_mask].to(DEVICE)

        logits_location, logits_type, logits_token = model(input_ids)
        location_pred = logits_location.argmax(dim=1)
        type_pred     = logits_type.argmax(dim=1)
        token_pred    = logits_token.argmax(dim=1)

        metas.extend(meta_data)
        preds.extend([{
            "predicted_location": token_index_to_char_index(code, tokens_, loc, meta),
            "predicted_type": FIX_TYPES[typ],
            "predicted_token": model.vocab[tok],
            "predicted_code": fix_code(
                code, token_index_to_char_index(code, tokens_, loc, meta),
                FIX_TYPES[typ], model.vocab[tok], 
                tokens_[loc] if loc < len(tokens_) else ""),
            "wrong_code": code,
            "metadata": meta
        } for loc, typ, tok, tokens_, code, meta
          in zip(location_pred, type_pred, token_pred, tokens, wrong_code, meta_data)
        ])
	
    for pred in preds:
        if pred["predicted_type"] == "delete":
            del pred["predicted_token"]

    n_correct = sum([is_correct(pred["predicted_code"]) for pred in tqdm(preds, desc="Checking correctness")])
    print(f"Code corrected: {n_correct}/{len(preds)} ({n_correct/len(preds)*100:.1f}%)")
    return preds


def fix_code(code: str, char_i: int, typ: str, tok: str, old_tok: str):
    if typ == "delete":
        return f"{code[:char_i]}{code[char_i+len(old_tok):]}"
    elif typ == "modify":
        return f"{code[:char_i]}{tok}{code[char_i+len(old_tok):]}"
    elif typ == "insert":
        return f"{code[:char_i]}{tok}{code[char_i:]}"


def load_model(source: str):
    model = torch.load(source)
    model.eval()
    return model


def write_predictions(destination: str, predictions: List[dict]):
    json.dump(
        sorted(predictions, key=lambda p: p["metadata"]["id"]), 
        open(destination, "w"), indent=2
    )


if __name__ == "__main__":
    args = parser.parse_args()
    args.destination = Path(args.destination)
    args.source = Path(args.source)
	
	# load the serialized model
    model = load_model(args.model)

	# predict incorrect location for each test example.
    predictions = predict(model, args.source)

	# write predictions to file
    write_predictions(args.destination, predictions)

	