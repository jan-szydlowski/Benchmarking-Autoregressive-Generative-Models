# Autoregressive Neural Networks for Boltzmann Distribution Sampling

The goal of this project is to investigate how different model architectures learn complex probability distributions and how they differ in terms of convergence, sample quality, computational cost, and training stability.

This project compares different autoregressive neural network architectures for sampling from the Boltzmann distribution.

The compared architectures are:

Variational Autoregressive Network (VAN)
PixelCNN
LSTM
Decoder-only Transformer

The project is implemented in PyTorch.

---
## Results

The models are evaluated using:

* Variational Free Energy (VFE)
* Effective Sample Size (ESS)
* Energy statistics
* Magnetization
* Training time
* Sampling time
* Number of trainable parameters

### Final comparison

| Model       | VFE | ESS | Parameters | Training Time | Sampling Time |
| ----------- | --: | --: | ---------: | ------------: | ------------: |
| VAN         | TBD | TBD |        TBD |           TBD |           TBD |
| PixelCNN    | TBD | TBD |        TBD |           TBD |           TBD |
| LSTM        | TBD | TBD |        TBD |           TBD |           TBD |
| Transformer | TBD | TBD |        TBD |           TBD |           TBD |

The table will be updated as the final experiments are completed.

---

## Example Visualizations

### Variational Free Energy

![VFE comparison](results/figures/vfe_comparison.png)

### Effective Sample Size

![ESS comparison](results/figures/ess_comparison.png)

### Generated Samples

![Generated samples](results/figures/generated_samples.png)

---

## Research Question

The main research question is:

> How does neural network architecture affect the ability of autoregressive generative models to learn and sample from a complex Boltzmann probability distribution?

The project focuses not only on final model quality, but also on trade-offs between accuracy, convergence speed and computational cost.

---

## Models

### VAN

A fully connected autoregressive model using masked linear layers.

The joint probability distribution is factorized as:

$$
q(x) = \prod_{i=1}^{N} q(x_i \mid x_1, \ldots, x_{i-1})
$$

Each variable is sampled sequentially from a conditional probability predicted by the network.

### PixelCNN

A convolutional autoregressive architecture using masked convolutions.

PixelCNN can exploit spatial relationships while preserving autoregressive ordering.

The implementation uses:

* Mask A in the first layer
* Mask B in subsequent layers
* convolutional feature channels
* autoregressive sampling

### LSTM

The configuration is represented as a sequence and generated one element at a time.

The LSTM uses previous generated values as input and predicts the probability of the next value.

The model can optionally receive additional positional information.

### Transformer

A decoder-only Transformer using causal self-attention.

The architecture contains:

* token embeddings
* positional embeddings
* causal self-attention
* feed-forward layers
* layer normalization
* linear prediction head

Causal masking prevents the model from accessing future values during autoregressive generation.

---

## Project Structure

```text
.
├── src/
│   ├── models/
│   │   ├── van.py
│   │   ├── pixelcnn.py
│   │   ├── lstm.py
│   │   └── transformer.py
│   │
│   ├── training/
│   │   └── trainer.py
│   │
│   ├── metrics/
│   │    └── metrics.py
│   │
│   └── physics/
│        └── energy.py
│
├── experiments/
│   └── configs/
│       ├── van.yaml
│       ├── pixelcnn.yaml
│       ├── lstm.yaml
│       └── transformer.yaml
│
├── notebooks/
│   └── analysis.ipynb
│
├── results/
│   ├── figures/
│   └── tables/
│   
│
├── train.py
├── requirements.txt
└── README.md
```

---

## Experimental Pipeline

Each experiment follows the same general pipeline:

```text
Configuration
      ↓
Model initialization
      ↓
Autoregressive sampling
      ↓
Loss calculation
      ↓
Backpropagation
      ↓
Metric calculation
      ↓
Saved results
      ↓
Final analysis
```

Keeping the training and evaluation pipeline shared between models makes the comparison more consistent.

---

## Configuration

Model settings are stored in YAML files.

Example:

```yaml
model: transformer

lattice_size: 8
beta: 0.3

training:
  epochs: 5000
  batch_size: 1024
  learning_rate: 0.001

model_params:
  embedding_dim: 64
  num_heads: 4
  num_layers: 3
```

This makes experiments reproducible without modifying the source code.

---

## Evaluation

### Variational Free Energy

The main optimization objective is:

$$
F_q =
\mathbb{E}_{x \sim q}
\left[
E(x) + \frac{1}{\beta}\log q(x)
\right]
$$

Lower VFE indicates a better approximation of the target distribution.

### Effective Sample Size

Sample quality is additionally evaluated using effective sample size:

$$
ESS =
\frac{
\left(\sum_i w_i\right)^2
}{
\sum_i w_i^2
}
$$

The normalized ESS is:

$$
ESS_{\text{norm}} = \frac{ESS}{N}
$$

where \(N\) is the number of samples.

### Additional Metrics

The project also tracks:

* energy
* magnetization
* convergence curves
* training time
* sampling time
* parameter count

---

## Analysis

The main experimental analysis is located in:

```text
notebooks/analysis.ipynb
```

The notebook contains:

1. experiment setup
2. result loading and validation
3. VFE convergence comparison
4. ESS comparison
5. energy analysis
6. magnetization analysis
7. generated sample visualization
8. computational performance comparison
9. final conclusions

The notebook is used for analysis and visualization only. Model implementation and training logic are kept in the source files.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

Example:

```bash
python train.py --config experiments/configs/transformer.yaml
```

Results are saved to:

```text
results/tables/
```

and generated plots to:

```text
results/figures/
```

---

## Reproducibility

To make the comparison reliable, models should be evaluated using comparable experimental conditions.

Important controls include:

* identical lattice size
* identical inverse temperature
* comparable training budgets
* fixed random seeds
* multiple independent training runs
* identical evaluation procedures

Where possible, results should be reported as:

```text
mean ± standard deviation
```

across multiple runs.

---

## Technologies

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook
* YAML
* Git

---

## Key Idea

This project is not only about determining which architecture achieves the lowest loss.

The main objective is to understand **why different autoregressive architectures behave differently** and what trade-offs they introduce in terms of sample quality, convergence and computational efficiency.
