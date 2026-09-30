# genpark-pairwise-bradley-terry-reward-model-skill

Bradley-Terry preference model implementation calculating logistic choice likelihoods, negative log-likelihood loss, and analytical gradient updates.

## Architecture

```mermaid
flowchart LR
    Scores["Chosen Score r_c & Rejected Score r_r"] --> Logit["Score Margin (r_c - r_r)"]
    Logit --> Sigmoid["Logistic Sigmoid Function"]
    Sigmoid --> Prob["P(y_c > y_r)"]
    Prob --> NLL["Binary Cross-Entropy Loss"]
```

## Features
- **Analytical Gradient Step**: Computes exact single-step reward shift.
- **Zero Dependencies**: 100% Python Standard Library.
