# Milestone 1

The task was to implement a tokenization procedure for python code.

The solution uses the `tokenize` package. Therefore it assumes the Python code to be correct (unlike tokenization in Milestone 2). From the output of `tokenize.tokenize` the following things are discarded: `[tokenize.ENCODING, tokenize.ENDMARKER, tokenize.ERRORTOKEN]` as well as all other tokens with an empty string value.

```python
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
```

# Milestone 2

The task was to train a model to predict the error location. 

The solution is splitted in [data.py](milestone2/data.py) and [Train.py](milestone2/Train.py) to handle data loading and modeling/training repsectively.

## Data Loading and Tokenization

The main challenge regarding data loading was to implement a tokenization procedure that is able to handle incorrect source code. 
The `tokenize.tokenize` function is throwing an error, if parantheses are unmatched or indentation is wrong. 
Luckily, in the case of a `TokenError` (mainly unmatched parantheses, occurs 9065 times in 50000 examples), the error is thrown at the end of the sequence, such that the procedure can be cut off without losing any information. In the case of the `IndentationError` (occurs 132 times in 50000 examples), the error is also thrown a few tokens after the actual error, but some tokens are cut off in the most cases. This is suboptimal, since it reduces the number error locations and therefore makes it easier for the model. On the other hand, this occurs only 132/50000 times, so the effect is marginal, which is why this is tolerated. The alternative would be to use the correct code for tokenization and apply the modifications afterwards, but this would make the model useless in real world applications, where the correct code is not available.
Another issue is, that for some instances the tokenization is not correct due to different python versions (Python version used for the original code and Python version 3.8 used in this project). Specifically this happens for 

- some walrus operators `:=` (assignment expression): `:==` gets tokenized to `[':=', '=']` instead of `[':', '==']`
- some decorators: `@==` gets tokenized to `['@=', '=']` instead of  `['@', '==']`

Those errors are also very rare (85 and 4 times out of 50000 instances respectively). They are detected by reconstructing the character index from the token index, as implemented in the main function of `data.py`.

A summary of all tokenization errors gets printed when executing `data.py`:

```
WARNING:root:TokenError           occurred 9065    of 50000   times at ID's 8, 9, 11, 20, 36, 39, 47, 50, 58, 68...
WARNING:root:IndentationError     occurred 132     of 50000   times at ID's 428, 493, 1428, 1598, 1836, 1866, 2660, 2706, 3036, 3381...
WARNING:root:walrus               occurred 85      of 50000   times at ID's 1800, 2637, 2754, 3174, 3782, 5708, 5816, 6094, 6966, 7999...
WARNING:root:decorator            occurred 4       of 50000   times at ID's 14066, 27131, 32921, 46163
```

Additionally to loading a single file into memory, the script offers an interface to use the whole dataset (by passing the directory instead of the file path as `--source` arg) rather than a single file. 

The `load_data` (or `load_multiple`) function returns the trainset, testset and the vocabulary.


## Modeling

The architecture is simple and mainly consists of a multi-layer `nn.LSTM`. The `input_ids` (index in vocabulary) are transformed to dense embeddings using a `nn.Embedding` layer.
The sequences are padded per batch and passed through the LSTM. The hidden states per token are further processed with a shared `nn.Linear` layer with a single output neuron. This output neuron indicates the likelihood of this token being the error location. The `argmax` over theses probabilities is the predicted error location.

The training procedure follows the standard Pytorch training loop. Train and test Loss and accuracy are logged to the `tqdm` progress bar. Evaluation is done once per epoch.

## Results

No formal hyperparameter tuning was performaed, but during trial and error the following hyperparameters have performed best:

```python
hparams = {
    "batch_size": 16,
    "bidirectional": True,
    "n_epochs": 80 if args.source.is_file() else 8,
    "emb_dim": 64,
    "num_layers": 5,
    "lr": 0.01
}
```

The achieved test accuracy is between 80% and 90% when training on a single file and between 90% and 95% when training on the whole dataset. Variations are due to random data splitting and parameter initialization. 


# Milestone 3

To also predict, the type of correction and the token to correct the error, 2 additional classification heads are attached to the architecture.
Both are only applied to the hidden representation of the predicted error location.

