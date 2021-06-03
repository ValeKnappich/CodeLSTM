# Milestone 1

The task was to implement a tokenization procedure for python code.

The solution uses the `tokenize` package. Therefore it assumes the Python code to be correct (unlike tokenization in Milestone 2). From the output of `tokenize.tokenize` the following things are discarded: `[tokenize.ENCODING, tokenize.ENDMARKER, tokenize.ERRORTOKEN]` as well as all other tokens with an empty string value.

```
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

Since the task is formulated as a token classification (see below) task, the given character index of the error location has to be transformed to the token index.
```
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
```

Additionally to loading a single file into memory, the script offers an interface to use the whole dataset (by passing the directory instead of the file path as `--source` arg) rather than a single file. 

The `load_data` (or `load_multiple`) function returns the trainset, testset and the vocabulary.


## Modeling

The architecture is simple and mainly consists of a multi-layer `nn.LSTM`. The `input_ids` (index in vocabulary) are transformed to dense embeddings using a `nn.Embedding` layer.
The sequences are padded per batch and passed through the LSTM. The hidden states per token are further processed with a shared `nn.Linear` layer with a single output neuron. This output neuron indicates the likelihood of this token being the error location. The `argmax` over theses probabilities is the predicted error location.

The training procedure follows the standard Pytorch training loop. Train and test Loss and accuracy are logged to the `tqdm` progress bar. Evaluation is done once per epoch.

## Results

