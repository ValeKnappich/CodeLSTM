---
title: ASDL Project
subtitle: Fixing syntactically incorrect code with Deep Learning
author: Valentin Knappich
date: \today
institute: University of Stuttgart, Analyzing Software using Deep Learning, Prof. Michael Pradel

theme: metropolis
slide_level: 1
aspectratio: 169
fontsize: 12pt
theme: metropolis

header-includes: |
    \usepackage{tabto}
    \NumTabs{10}

    \definecolor{primary}{HTML}{384D48}
    \definecolor{design}{HTML}{ACAD94}
    \setbeamercolor{palette primary}{fg=white, bg=primary}
    \setbeamercolor{progress bar}{fg=primary, bg=design}
    \setbeamercolor{background canvas}{bg=white}

    \metroset{numbering=fraction, progressbar=foot}

    `\setbeamertemplate{section in toc}[sections numbered]`{=latex}
    `\setbeamertemplate{subsection in toc}[subsections numbered]`{=latex}

    \makeatletter
    \setlength{\metropolis@titleseparator@linewidth}{1.5pt}
    \setlength{\metropolis@progressonsectionpage@linewidth}{1.5pt}
    \setlength{\metropolis@progressinheadfoot@linewidth}{1.5pt}
    \makeatother

    \usepackage{listings}

    \definecolor{background}{RGB}{255, 255, 255}
    \definecolor{string}{RGB}{230, 219, 116}
    \definecolor{comment}{RGB}{117, 113, 94}
    \definecolor{normal}{RGB}{248, 248, 242}
    \definecolor{identifier}{RGB}{166, 226, 46}

    \lstset{
        language=python,                			% choose the language of the code
        stepnumber=1,                   		% the step between two line-numbers.        
        numbersep=5pt,                  		% how far the line-numbers are from the code
        numberstyle=\tiny\color{black}\ttfamily,
        backgroundcolor=\color{background},  		% choose the background color. You must add \usepackage{color}
        showspaces=false,               		% show spaces adding particular underscores
        showstringspaces=false,         		% underline spaces within strings
        showtabs=false,                 		% show tabs within strings adding particular underscores
        tabsize=4,                      		% sets default tabsize to 2 spaces
        captionpos=b,                   		% sets the caption-position to bottom
        breaklines=true,                		% sets automatic line breaking
        breakatwhitespace=true,         		% sets if automatic breaks should only happen at whitespace
        title=\lstname,                 		% show the filename of files included with \lstinputlisting;
        basicstyle=\color{normal}\ttfamily,					% sets font style for the code
        keywordstyle=\color{magenta}\ttfamily,	% sets color for keywords
        stringstyle=\color{string}\ttfamily,		% sets color for strings
        commentstyle=\color{comment}\ttfamily,	% sets color for comments
        emph={},
        emphstyle=\color{identifier}\ttfamily
    }
---

## Agenda

\tableofcontents

# Preprocessing

## Tokenization

- `tokenize` package
- Error handling needed for incorrect code: 
    - `TokenError`
        - Thrown at the end of the sequence \textrightarrow no tokens lost
    - `IndentationError`
        - Sometimes thrown before the end of sequence \textrightarrow tokens lost
        - Advantage for the model
        - Occurs only 132 times in the whole dataset (50000 samples)

## Testing Preprocessing

- Testing tokenization via reconstruction
    - Run tokenization
    - Convert character index to token index
    - Convert token index back to character index
    \item[\textrightarrow] Misalignments?
- Walrus operator `:=` (fails 85 times in 50000 samples)
    - Was introduced in Python 3.8 
    - `":=="` gives `[":=", "="]` instead of `[":", "=="]`
- Decorator `@` (fails 4 times in 50000 samples)
    - `"@=="` gives `["@=", "="]` instead of `["@", "=="]`


# Modeling & Training

## Architecture

\begin{figure}
    \centering
    \includegraphics[width=.9\linewidth]{img/arch0.png}
\end{figure}

## Architecture

\begin{figure}
    \centering
    \includegraphics[width=.9\linewidth]{img/arch1.png}
\end{figure}

## Architecture

\begin{figure}
    \centering
    \includegraphics[width=.9\linewidth]{img/arch2.png}
\end{figure}

## Training

- `Adam` optimizer and `CrossEntropyLoss`
- Multi-task training (adding losses together)
- Problem: As long as the location prediction is bad, the signal from type and token prediction are just irritating
- Solution: Use linear loss weighting schedule
    - decrease location loss weight
    - increase type and token loss weight
\vspace{15pt}

\begin{figure}
\centering
\begin{lstlisting}[language=Python, basicstyle=\scriptsize, xleftmargin=.1\textwidth]
location_weight = torch.tensor([-(x + 1) / n_epochs + 1 
                                for x in range(n_epochs)])
type_weight     = torch.tensor([(x + 1) / n_epochs 
                                for x in range(n_epochs)])
token_weight    = torch.tensor([(x + 1) / n_epochs 
                                for x in range(n_epochs)])
\end{lstlisting}
\end{figure}


# Results

## Results

- Results vary depending on random initialization and random test split
- Evaluating on test set once per epoch
    - Location Accuracy: $~97\%$
    - Fix Type Accuracy: $~89\%$
    - Fix Token Accuracy: $~89\%$
- Prediction
    - Fraction of corrected code snippets: $~92\%$

## Questions

\begin{figure}
\centering
\includegraphics[width=.4\linewidth]{img/question}
\end{figure}