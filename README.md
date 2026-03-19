## Numpy Word2Vec: Skip-Gran with Negative Sampling (SGNS)
This repository contains a pure **Numpy** implementation of the Word2Vec algorithm, specifically
the **Skip-Gram** arquitecture with **Negative Sampling**. This project was developed to
demonstrate a deep understanding of word embedding theory, including forward propagation, custom
gradient derivation and efficiency in training loops.

## Overview
The goal of this implementation is to map words from a discrete space to a continuous vector space
where semantically similar words are positioned close to each other.

## Key Features:

* **Architecture**: Skip-Gram (predicting context words given a center word).
* **Optimization**: Negative Sampling (SGNS) for efficient training on large vocabularies.
* **Subsampling**: Frequent word subsampling to reduce bias and speed up training.
* **Efficiency**: Vectorized operations using `numpy.einsum` and `np.add.at` for high performance.
* **Stability**: Numerical stability tricks like sigmoid clipping and epsilon-buffered logs.

## Project Structure

* `model.py`: The core Word2Vec class containing the forward pass, loss calculation, and manual backpropagation.
* `train.py`: Training pipeline including the learning rate scheduler (linear decay).
* `preprocessing.py`: Text cleaning, vocabulary building, and unigram table generation for negative sampling.
* `eval.py`: A CLI tool to test word similarities using cosine distance.
* `environment.yml`: Conda environment specification.

## Mathematical Implementation

The core logic of this project resides in `model.py`, where the Skip-Gram objective is optimized using Negative Sampling.

### Forward Pass & Loss
For a center word $w$ and context word $c$, the model maximizes the probability of the context word appearing near the center word. To avoid the computational expense of a full Softmax, we use **Negative Sampling**. 

The objective function for a single pair with $k$ negative samples is:
$$J = -\log \sigma(v_c^\top v_w) - \sum_{i=1}^k \log \sigma(-v_{n_i}^\top v_w)$$

* **Sigmoid Function**: Implemented with numerical clipping to $[-6, 6]$ to prevent overflow/underflow in `np.exp`.
* **Loss Calculation**: A small epsilon ($10^{-10}$) is added to the log terms to ensure numerical stability.

### Backpropagation (Gradients)
Gradients are calculated manually and updated using the learning rate $\eta$:

1.  **Output Vectors ($W_2$):**
    * **Positive context**: $\frac{\partial J}{\partial v_c} = (\sigma(v_c^\top v_w) - 1) v_w$
    * **Negative samples**: $\frac{\partial J}{\partial v_n} = \sigma(v_n^\top v_w) v_w$
2.  **Input Vector ($W_1$):**
    * $\frac{\partial J}{\partial v_w} = (\sigma(v_c^\top v_w) - 1) v_c + \sum_{i=1}^k \sigma(v_{n_i}^\top v_w) v_{n_i}$

**Key Implementation Detail**: We use `np.add.at` for all weight updates. This is critical because frequent words may appear multiple times within a single batch; `np.add.at` ensures that gradients are **accumulated** rather than overwritten.

---

## Getting Started

### 1. Requirements
The project requires Python 3.10 and NumPy. You can recreate the environment using the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate word2vec_numpy
```

### 2. Dataset
This implementation is designed to work with the **text8** dataset (the first 100MB of cleaned English Wikipedia dump).
1.  Create a `data/` folder in the project root.
2.  Download the [text8 dataset](http://mattmahoney.net/dc/text8.zip).
3.  Place the unzipped `text8` file inside the `data/` folder.
4.  The `preprocessing.py` script will automatically locate it at `../data/text8`.

### 3. Training
To start the training process, execute the training script:
```bash
python train.py
```

### 4. Evaluation

Test the learned embeddings by searching for similar words:

```bash
python eval.py
```

## Optimization Highlights 

* **Unigram Table**: Used for $P(w)^{3/4}$ sampling, ensuring rare words are sampled more frequently than their raw distribution.
* **Dynamic Window**: Implementation of a variable context window size to capture different levels of semantic relationships.
* **Memory Management**: Batch generation uses a buffered approach to handle large text files without exhausting RAM.
