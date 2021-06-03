import argparse
from pathlib import Path
import json
from typing import List

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data import combine_batch, load_data, load_multiple, token_index_to_char_index
from Train import CodeLSTM      # 'useless' import is needed for torch to load model


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', help="Path of trained model.", default="model.pth")
parser.add_argument(
    '--source', help="Folder path of all test files.", default="dataset")
parser.add_argument(
    '--destination', help="Path to output json file of extracted predictions.", default="predictions.json")


def predict(model: torch.nn.Module, test_files: str):
    # Enable loading multiple files
    if test_files.is_file():
        (train_ds, _), _ = load_data(test_files, test_frac=0)
    elif test_files.is_dir():
        (train_ds, _), _ = load_multiple(test_files, test_frac=0)
    else:
        raise FileNotFoundError(f"File or directory {test_files.absolute()} was not found")

    dl = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=combine_batch)
    preds, metas, char_preds = [], [], []
    for tokens, labels, meta_data, wrong_code in tqdm(dl):
        labels = torch.tensor(labels)
        logits = model(tokens)
        preds_ = logits.argmax(dim=1)

        preds.extend(preds_.tolist())
        metas.extend(meta_data)
        char_preds.extend([
            token_index_to_char_index(code, tokens_, pred, meta)
            for code, tokens_, pred, meta in zip(wrong_code, tokens, preds_, meta_data)
        ])
	
    return [
        {"predicted_location": pred, "metadata": meta}
        for pred, meta in zip(char_preds, metas)
    ]


def load_model(source: str):
    model = torch.load(source)
    model.eval()
    return model


def write_predictions(destination: str, predictions: List[dict]):
    n_correct = sum([
        1 if prediction["predicted_location"] == prediction["metadata"]["fix_location"] else 0
        for prediction in predictions
    ])
    print(f"Test Accuracy: {n_correct / len(predictions)}")
    json.dump(predictions, open(destination, "w"), indent=2)


if __name__ == "__main__":
    args = parser.parse_args()
    args.destination = Path(args.destination)
    args.source = Path(args.source)
	
	# load the serialized model
    model = load_model(args.model)

	# predict incorrect location for each test example.
    predictions = predict(model, args.source)

	# write predictions to file
    write_predictions(args.destination,predictions)

	