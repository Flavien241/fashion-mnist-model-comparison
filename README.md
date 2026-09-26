# Fashion-MNIST: MLP vs. CNN

A reproducible PyTorch experiment comparing a multilayer perceptron with a LeNet-inspired convolutional neural network for Fashion-MNIST image classification.

## Highlights

- held-out train/validation/test protocol;
- Adam optimisation and cross-entropy loss;
- fixed random seeds for repeatable data splits and training;
- automatic comparison of test accuracy between the MLP baseline and CNN.

## Run

```bash
pip install -r requirements.txt
python tp2_cnn_fashionmnist.py --epochs 5 --seed 42
```

Fashion-MNIST is downloaded automatically on first run. The script prints the loss and accuracy of both models and their test-accuracy difference.

## Additional material

- `assets/` contains dataset visualisations used in the coursework;
- `supplementary-mlp/` contains the accompanying PyTorch MLP notebook and small helper scripts.

## Context

Academic coursework completed at Polytech Lyon. This repository is presented as a compact, reproducible supervised-learning experiment, not as a novel method.
