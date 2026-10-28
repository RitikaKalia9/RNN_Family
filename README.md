# Recurrent Neural Networks from First Principles

RNN · BPTT · LSTM · GRU · BiLSTM in PyTorch. Everything runs on a CPU in under a minute, and no datasets are needed because every sequence is generated inside the notebook.

## What's inside

| Part | Topic | What it shows |
| --- | --- | --- |
| 1 | RNN | Forward pass by hand, a cell built from `nn.Linear`, and why the parameter count does not depend on sequence length |
| 2 | BPTT | Gradients through time derived by hand and verified against autograd and a finite difference; vanishing and exploding gradients; gradient clipping; truncated BPTT |
| 3 | LSTM | Gates traced by hand, the effect of the forget-gate bias on gradient reach, and a 60-step memory task |
| 4 | GRU | A textbook GRU rebuilt against `nn.GRUCell`, and parameter counts for every cell, with and without bidirectionality |
| 5 | BiLSTM | A BiLSTM shown to be two LSTMs, then tested on a task that needs the next token and on a simulated live stream |

## Key results

| Experiment | Result |
| --- | --- |
| Scalar BPTT (hand / autograd / finite difference) | All three give 0.013816 |
| Matrix BPTT | Hand-written gradients match autograd for `W_xh`, `W_hh` and `W_hy` |
| Gradient reach at T=40 (plain RNN, default weights) | 8.64e-12 (vanishes) |
| Gradient reach at T=40 (recurrent matrix scaled by 3) | 2.40e+00 (grows) |
| Truncated vs full BPTT | Cosine similarity 0.9945 |
| 60-step recall: RNN / LSTM default / LSTM forget bias 2 | 68.9% / 51.0% / 100.0% |
| 60-step recall: GRU default / GRU update-gate bias 2 | 51.4% / 100.0% |
| Next-token tagging: LSTM / BiLSTM / BiLSTM on a stream | 50.0% / 100.0% / 49.0% |
| Previous-token tagging: LSTM / BiLSTM / BiLSTM on a stream | 100.0% / 100.0% / 100.0% |

## Takeaways

- Gradients of shared weights are sums over time steps, and a hand-written BPTT must match autograd.
- Gradients through a plain RNN shrink or grow geometrically. Clipping fixes the growth but not the shrinkage.
- An LSTM's additive cell state gives gradients a path, but the forget-gate values decide whether it is open. A forget bias of 2 took the recall task from 51% to 100%.
- The textbook GRU and `nn.GRU` differ in two places: what the update gate weights, and where the reset gate is applied.
- `out[:, -1]` of a BiLSTM is not the sequence representation. Use `torch.cat([h_n[0], h_n[1]], dim=1)`.
- A BiLSTM needs the whole sequence in advance. It scored 100% offline and 49% on a stream, with the same weights.

## Parameter counts (d_x = 10, d_h = 20)

Formula: `P = k · d_h · (d_x + d_h + 1)`, with k = 1 (RNN), 3 (GRU), 4 (LSTM). Bidirectional doubles the count. PyTorch reports `k · d_h` more per direction because it keeps two bias vectors.

| Model | Formula | PyTorch |
| --- | --- | --- |
| RNN | 620 | 640 |
| BiRNN | 1240 | 1280 |
| GRU | 1860 | 1920 |
| BiGRU | 3720 | 3840 |
| LSTM | 2480 | 2560 |
| BiLSTM | 4960 | 5120 |

## Requirements

- Python 3.9+
- PyTorch (`pip install torch`)

## Running it

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install torch jupyter
jupyter notebook rnn_family.ipynb
```

Run the cells in order, since later cells reuse names defined earlier (`reach`, `make_rnn`, `lstm_fb`, `make_batch`, `Recall`, `Tagger`). Seeds are fixed, but exact accuracies and gradient values can differ slightly across PyTorch versions. Expect the same pattern rather than the same last digit.

## Repository structure

```
.
├── rnn_family.ipynb
└── README.md
```
