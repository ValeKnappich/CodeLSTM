import argparse
import logging
from pathlib import Path
from typing import List

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data import load_data, load_multiple, combine_batch, is_correct

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Folder path of all training files.", default="../dataset/train")
parser.add_argument(
    '--destination', help="Path to save your trained model.", default="model.pth")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CodeLSTM(nn.Module):
    def __init__(
            self, vocab: list, emb_dim: int, num_layers: int, 
            bidirectional: bool, n_epochs: int, **kwargs
        ):
        super().__init__()
        self.vocab = vocab
        # Token order: PAD, UNK, vocab[0], ...
        self.num_token_ids = len(vocab) + 2
        self.embedd = nn.Embedding(self.num_token_ids, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, emb_dim, num_layers=num_layers, 
            batch_first=True, bidirectional=bidirectional
        )
        emb_factor = 2 if bidirectional else 1
        self.linear_location = nn.Linear(emb_factor * emb_dim, 1)
        self.linear_type     = nn.Linear(emb_factor * emb_dim, 3)
        self.linear_token    = nn.Linear(emb_factor * emb_dim, len(vocab))

        # Weight gradient update, start with high weight on location and move to typ and tok
        self.location_weight = torch.tensor([-(x + 1) / n_epochs + 1 for x in range(n_epochs)])
        self.type_weight     = torch.tensor([(x + 1) / n_epochs for x in range(n_epochs)])
        self.token_weight    = torch.tensor([(x + 1) / n_epochs for x in range(n_epochs)])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size  = input_ids.shape[0]
        input_emb   = self.embedd(input_ids) # get dense embeddings for ids
        lstm_out, _ = self.lstm(input_emb) # shape batch_size, seq_len, emb_dim

        logits_location = self.linear_location(lstm_out).view(batch_size, -1)
        pred_location   = logits_location.argmax(dim=1)
        selected_hidden = torch.stack([data[location,:] for data, location in zip(lstm_out, pred_location)])
        logits_token    = self.linear_token(selected_hidden)
        logits_type     = self.linear_type(selected_hidden)
        return logits_location, logits_type, logits_token


def acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return ((preds == labels).sum() / len(labels)).item()


def train_model(
        model: nn.Module, train_ds: List[dict], test_ds: List[dict], 
        n_epochs: int, batch_size: int, lr: float, **kwargs
    ):

    train_dl  = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=combine_batch)
    test_dl   = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=combine_batch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(DEVICE)

    # Initialize metrics, -1 meaning not yet computed
    train_accs, train_acc = ([], (-1, -1, -1))
    test_accs, test_acc = ([], (-1, -1, -1))
    progress_bar = tqdm(total=len(train_dl), desc="Training")
    for epoch in range(n_epochs):
        # Use same progressbar for each epoch
        progress_bar.refresh()
        progress_bar.reset()

        # Training loop
        model.train()
        for input_ids, _, fix_location, fix_type, fix_token, _, _ in train_dl:
            input_ids      = input_ids.to(DEVICE)
            fix_location   = fix_location.to(DEVICE)
            fix_type       = fix_type.to(DEVICE)
            fix_token_mask = fix_token != -1        # filter out the instances where there is no token to predict (delete type)
            fix_token      = fix_token[fix_token_mask].to(DEVICE)
            
            optimizer.zero_grad()
            logits_location, logits_type, logits_token = model(input_ids)
            logits_token = logits_token[fix_token_mask]
            loss = model.location_weight[epoch] * criterion(logits_location, fix_location) + \
                   model.type_weight[epoch]     * criterion(logits_token, fix_token) + \
                   model.token_weight[epoch]    * criterion(logits_type, fix_type)

            loss.backward()
            optimizer.step()

            train_acc_step = (
                acc(logits_location, fix_location),
                acc(logits_type, fix_type),
                acc(logits_token, fix_token)
            )
            train_accs.append(train_acc_step)

            progress_bar.set_description(f"Epoch {epoch}")
            progress_bar.set_postfix_str(
                f"train_acc (LOC, TYP, TOK): ({train_acc[0]:.2f}, {train_acc[1]:.2f}, {train_acc[2]:.2f}), "
                f"test_acc (LOC, TYP, TOK): ({test_acc[0]:.2f}, {test_acc[1]:.2f}, {test_acc[2]:.2f}), "
                f"train_loss_step: {loss.item():.2f}"
            )
            progress_bar.update()
        
        # Eval loop
        model.eval()
        for input_ids, _, fix_location, fix_type, fix_token, _, _ in test_dl:
            input_ids      = input_ids.to(DEVICE)
            fix_location   = fix_location.to(DEVICE)
            fix_type       = fix_type.to(DEVICE)
            fix_token_mask = fix_token != -1        # filter out the instances where there is no token to predict (delete type)
            fix_token      = fix_token[fix_token_mask].to(DEVICE)

            logits_location, logits_type, logits_token = model(input_ids)
            logits_token = logits_token[fix_token_mask]

            test_acc_step = (
                acc(logits_location, fix_location),
                acc(logits_type, fix_type),
                acc(logits_token, fix_token)
            )
            test_accs.append(test_acc_step)
        
        train_acc = (
            sum([acc[0] for acc in train_accs]) / len(train_accs),
            sum([acc[1] for acc in train_accs]) / len(train_accs),
            sum([acc[2] for acc in train_accs]) / len(train_accs),
        )
        test_acc = (
            sum([acc[0] for acc in test_accs]) / len(test_accs),
            sum([acc[1] for acc in test_accs]) / len(test_accs),
            sum([acc[2] for acc in test_accs]) / len(test_accs),
        )
        progress_bar.set_postfix_str(
            f"train_acc (LOC, TYP, TOK): ({train_acc[0]:.2f}, {train_acc[1]:.2f}, {train_acc[2]:.2f}), "
            f"test_acc (LOC, TYP, TOK): ({test_acc[0]:.2f}, {test_acc[1]:.2f}, {test_acc[2]:.2f}), "
            f"train_loss_step: {loss.item():.2f}"
        )


def save_model(model: nn.Module, destination: Path):
    if destination.exists():
        logging.warning(f"File {destination} exists, overwriting!")
    torch.save(model, destination)
	

if __name__ == "__main__":
    args = parser.parse_args()
    args.source = Path(args.source)
    args.destination = Path(args.destination)
    
    if args.source.is_file():
        (train_ds, test_ds), vocab = load_data(args.source)
    elif args.source.is_dir():
        (train_ds, test_ds), vocab = load_multiple(args.source)
    else:
        raise FileNotFoundError(f"File or directory {args.source.absolute()} was not found")

    hparams = {
        "batch_size": 32,
        "bidirectional": True,
        "n_epochs": 80 if args.source.is_file() else 8,
        "emb_dim": 64,
        "num_layers": 5,
        "lr": 0.001
    }

    model = CodeLSTM(vocab, **hparams)
    train_model(model, train_ds, test_ds, **hparams)
    save_model(model, args.destination)