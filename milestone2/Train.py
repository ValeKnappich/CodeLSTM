#!/usr/bin/python3
import argparse
import logging
from pathlib import Path
from typing import List

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data import load_data, load_multiple, combine_batch

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Folder path of all training files.", default="dataset/training_0.json")
parser.add_argument(
    '--destination', help="Path to save your trained model.", default="model.pth")


class CodeLSTM(nn.Module):
    def __init__(
            self, vocab: set, emb_dim: int, num_layers: int, 
            bidirectional: bool, **kwargs
        ):
        super().__init__()
        self.vocab = vocab
        # Token order: PAD, UNK, vocab[0], ...
        self.num_token_ids = len(vocab) + 2
        self.embedd = nn.Embedding(self.num_token_ids, emb_dim)
        self.token_to_id = {token: id + 2 for id, token in enumerate(self.vocab)}
        self.lstm = nn.LSTM(
            emb_dim, emb_dim, num_layers=num_layers, 
            batch_first=True, bidirectional=bidirectional
        )
        self.linear = nn.Linear(emb_dim, 1)

    def forward(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = len(batch_tokens)
        # Get input ids,
        input_ids = [
            torch.tensor([self.token_to_id.get(token, 1) for token in tokens])
            for tokens in batch_tokens
        ]
        input_ids = pad_sequence(input_ids, batch_first=True)
        input_emb = self.embedd(input_ids) # get dense embeddings for ids
        lstm_out, _ = self.lstm(input_emb) # shape batch_size, seq_len, emb_dim
        logits = self.linear(lstm_out).view(batch_size, -1)
        return logits


def acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return ((preds == labels).sum() / len(labels)).item()


def train_model(
        model: nn.Module, train_ds: List[dict], test_ds: List[dict], 
        n_epochs:int, batch_size: int, **kwargs
    ):

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=combine_batch)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=combine_batch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Initialize metrics, -1 meaning not yet computed
    train_accs, train_acc = ([], -1)
    test_accs, test_acc, test_loss = ([], -1,  torch.tensor(-1.))
    progress_bar = tqdm(total=len(train_dl))
    for epoch in range(n_epochs):
        model.train()
        # Use same progressbar for each epoch
        progress_bar.refresh()
        progress_bar.reset()

        # Training loop
        for tokens, labels, _, _ in train_dl:
            labels = torch.tensor(labels)
            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_acc_step = acc(logits, labels)
            train_accs.append(train_acc_step)

            progress_bar.set_description(f"Epoch {epoch}")
            progress_bar.set_postfix_str(
                f"train_acc_epoch: {train_acc:.2f}, train_acc_step: {train_acc_step:.2f}, test_acc: {test_acc:.2f} "
                f"train_loss: {loss.item():.2f}, test_loss: {test_loss.item():.2f}"
            )
            progress_bar.update()
        
        # Eval loop
        model.eval()
        for tokens, labels, _, _ in test_dl:
            labels = torch.tensor(labels)
            logits = model(tokens)
            test_loss = criterion(logits, labels)
            test_acc_step = acc(logits, labels)
            test_accs.append(test_acc_step)
        
        train_acc = sum(train_accs) / len(train_accs)
        test_acc = sum(test_accs) / len(test_accs)
        progress_bar.set_postfix_str(
            f"train_acc_epoch: {train_acc:.2f}, train_acc_step: {train_acc_step:.2f}, test_acc: {test_acc:.2f} "
            f"train_loss: {loss.item():.2f}, test_loss: {test_loss.item():.2f}"
        )


def save_model(model: nn.Module, destination: str):
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
        "batch_size": 8,
        "bidirectional": False,
        "n_epochs": 12,
        "emb_dim": 128,
        "num_layers": 3
    }

    model = CodeLSTM(vocab, **hparams)
    train_model(model, train_ds, test_ds, **hparams)
    save_model(model, args.destination)