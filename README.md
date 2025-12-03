# Next Pitch Prediction in Major League Baseball Using Transformer Models

## Author: Liam Bowen
## School: Vanderbilt University
## Course: Generative AI Models in Theory & Practice

## Summary

This repository contains a complete, end-to-end pipeline for predicting the next pitch type in Major League Baseball using Transformer sequence models.

Pitch sequences are treated like language tokens. The model learns:

```math
P(\text{next pitch} \mid \text{previous pitches + context})
```

## Table of Contents
- [Overview](#overview)
- [Methodology](#methodology)
- [Implementation & Demo](#implementation)
- [Assessment &Evaluation](#evaluation)
- [Model & Data Cards](#model--data-cards)
- [Critical Analysis](#critical-analysis)
- [Resources](#resources)

## Overview

Major League Baseball generates millions of pitches each season, yet predicting what pitch is coming next will always be one of the hardest problems in baseball analytics. Understanding pitch sequencing has value for:
- Hitters and coaches preparing for opponents
- Broadcasters explaining strategy
- Betting and forecasting models
- Player development and scouting

### Problem Statement:
Can we accurately predict the next pitch type thrown in an MLB at-bat using modern sequence-modeling techniques, specifically Transformer architectures. How does this approach compare to simpler baselines such as logistic regression?

This project builds a complete, end-to-end system for MLB pitch prediction, including data preprocessing, pitch tokenization, a logistic regression baseline, a custom Transformer model, uncertainty calibration, and multiple evaluation analyses.

## Methodology
This project follows a structured machine learning pipeline:

1. Data Preparation
  - Loaded MLB pitch-by-pitch data
  - Filtered and cleaned variables relevant for pitch sequencing:
      - pitch type
      - count (balls/strikes)
      - pitcher ID
      - contextual features (score, inning, first pitch of at bat, etc.)
  - Mapped pitch types to integer tokens (FF -> 5, SL -> 15)
   
2. Baseline Model: Multiclass Logistic Regression
  - Inputs:
      - Pitch count (balls/strikes)
      - Game state (score differential)
  - Outputs: probabilites over 17 pitch types
  - Evaluated using:
      - Top-1 Accuracy
      - Top-3 Accuracy
      - Brier Score
      - Expected Calibration Error (ECE)

3. Transformer Pitch-Sequence Model
  - Custom PyTorch implementation
  - Encodes pitch history for each pitcher as a sequence
  - Uses embeddings for:
      - Pitch tokens
      - Balls
      - Strikes
      - First-pitch flag
      - Pitcher ID
  - 4-head self-attention
  - Sequence length = 32 (last 32 pitches thrown by that pitcher)
  - Output: softmax over 17 pitch types

4. Model Calibration
  Transformers often output overconfident probabilities, so we apply:
    - Temperature scaling
    - Reliability diagrams
    - Before/after ECE & Brier score comparison

5. Evaluation
   We conduct a wide range of analyses:
     - Top-k accuracy
     - Pitch-type-specific accuracy
     - Count-based accuracy heatmaps
     - Per-pitcher model performance
     - Calibration curves
  

## Implementation & Demo

This section demonstrates how to load the model, run predictions, and visualize calibration.

1. Load Model & Data
```python
import torch
from model import PitchTransformer
from data import load_test_set

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PitchTransformer.load_from_checkpoint("checkpoints/transformer_best.pt")
model = model.to(device).eval()

X_test, y_test = load_test_set()

```

2. Generate Predictions
```python
with torch.no_grad():
    logits = model(X_test.to(device))
    probs  = torch.softmax(logits, dim=1)
    preds  = probs.argmax(dim=1).cpu().numpy()
```

3. Example Prediction
```python
sample = 42
print("Previous pitches:", X_test['pitch_tokens'][sample])
print("True next pitch:", y_test[sample])
print("Predicted:", preds[sample])
print("Probabilities:", probs[sample])
```

Example output:
```less
Previous pitches: ['FF', 'SL', 'SL']
True next pitch: FF
Predicted: FF
Probabilities: [0.02, 0.45, 0.38, ...]
```

4. Compute Test Accuracy
```python
import numpy as np

acc = np.mean(preds == y_test.numpy())
print(f"Transformer Test Accuracy: {acc:.3f}")
```

5. Reliability Curve
```python
from metrics import reliability_curve
import matplotlib.pyplot as plt

centers, avg_conf, avg_acc = reliability_curve(y_test, probs.cpu().numpy())

plt.plot([0,1],[0,1],"--", label="Perfect Calibration")
plt.plot(avg_conf, avg_acc, marker="o", label="Transformer")
plt.xlabel("Predicted probability")
plt.ylabel("Empirical accuracy")
plt.legend()
plt.title("Reliability Diagram")
plt.show()
``` 

## Assessment & Evaluation

This project evaluates both baseline and Transformer models using:

1. Overall Accuracy
2. Top-k Accuracy
3. Calibration Metrics
   - Brier Score
   - Expected Calibration Error (ECE)
   - Temperature Scaling
4. Per-Pitch-Type Accuracy
   Shows which pitches (FF, SL, CU, CH, etc.) are easiest/hardest to predict
5. Count-Based Heatmap
   Visualizes how model accuracy changes in different ball-strike counts


## Model & Data Cards







