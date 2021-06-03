#!/usr/bin/python3
import argparse
from pathlib import Path
from unicodedata import bidirectional
from test.support import transient_internet

import torch
from torch import nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data import load_data, load_multiple

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Folder path of all training files.", default="dataset/training_0.json")
parser.add_argument(
    '--destination', help="Path to save your trained model.", default="model.pth")


class CodeLSTM(nn.Module):
    def __init__(self, vocab, emb_dim, num_layers, bidirectional, **kwargs):
        super().__init__()
        self.vocab = vocab
        self.embedd = nn.Embedding(len(vocab) + 1, emb_dim) # all tokens plus padding token
        self.token_to_id = {token: id + 1 for id, token in enumerate(self.vocab)}
        self.num_token_ids = len(vocab) + 1
        self.lstm = nn.LSTM(
            emb_dim, emb_dim, num_layers=num_layers, 
            batch_first=True, bidirectional=bidirectional
        )
        self.linear = nn.Linear(emb_dim, 1)

    def forward(self, batch_tokens):
        batch_size = len(batch_tokens)
        input_ids = [
            torch.tensor([self.token_to_id[token] for token in tokens])
            for tokens in batch_tokens
        ]
        input_ids = pad_sequence(input_ids, batch_first=True)
        input_emb = self.embedd(input_ids)
        lstm_out, _ = self.lstm(input_emb) # bs, sq_len, emb_dim
        logits = self.linear(lstm_out).view(batch_size, -1)
        return logits


def combine_batch(batch):
    # Transform from individual dicts to combined lists
    batch_tokens = [instance["tokens"] for instance in batch]
    batch_labels = [instance["error_index"] for instance in batch]
    return batch_tokens, batch_labels


def acc(logits, labels):
    preds = logits.argmax(dim=1)
    return ((preds == labels).sum() / len(labels)).item()


def train_model(model, train_ds, test_ds, n_epochs, batch_size, **kwargs):
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=combine_batch)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=combine_batch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_accs = []
    train_acc = -1
    test_accs = []
    test_acc = -1
    test_loss = torch.tensor(-1.)
    progress_bar = None
    for epoch in range(n_epochs):
        model.train()
        if not progress_bar:
            progress_bar = tqdm(total=len(train_dl))
        else:
            progress_bar.refresh()
            progress_bar.reset()
        for tokens, labels in train_dl:
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
        
        model.eval()
        for tokens, labels in test_dl:
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


def save_model(model, destination, hparams):
	"""
	TODO: Implement your method for saving the training the model here.
	"""
	raise Exception("Method save_model not implemented.")
	

if __name__ == "__main__":
    args = parser.parse_args()
    args.source = Path(args.source)
    if args.source.is_file():
        (train_ds, test_ds), vocab = load_data(args.source)
    elif args.source.is_dir():
        (train_ds, test_ds), vocab = load_multiple(args.source)
    else:
        raise FileNotFoundError(f"File or directory {args.source.absolute()} was not found")

    hparams = {
        "batch_size": 8,
        "bidirectional": False,
        "n_epochs": 20,
        "emb_dim": 128,
        "num_layers": 3
    }

    model = CodeLSTM(vocab, **hparams)
    train_model(model, train_ds, test_ds, **hparams)
    save_model(model, args.destination, hparams)
