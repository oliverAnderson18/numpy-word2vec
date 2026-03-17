import os
import numpy as np
from collections import Counter
import random

# Define relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/text8')

def load_text(file_path=DATA_PATH):
    """Reads the text file and returns a list of tokens."""
    
    print(f"Loading data from: {file_path}\n")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    return text.split()

def build_vocab(words, min_freq=5):
    """
    Builds vocabulary by filtering rare words.
    Returns mapping dictionaries and word frequencies.
    """
    print("Building vocabulary...\n")
    word_counts = Counter(words)
    
    vocab = [word for word, count in word_counts.items() if count >= min_freq]
    vocab = ["<UNK>"] + vocab
    
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for i, word in enumerate(vocab)}
    
    total_words = sum(word_counts.values())
    word_freqs = {word: count / total_words for word, count in word_counts.items()}
    
    return word2idx, idx2word, word_freqs
    

def subsample_text(words, word2idx, word_freqs, threshold=1e-3):
    """
    Applies subsampling to frequent words based on Mikolov's formula.
    Returns the text converted to numerical indices.
    """
    subsampled = []
    
    for word in words:
        idx = word2idx.get(word, word2idx['<UNK>'])
        freq = word_freqs.get(word, 0)
        if freq > threshold:
            p_keep = (np.sqrt(threshold / freq) + (threshold / freq))
        else:
            p_keep = 1.0
    
        if random.random() < p_keep:
            subsampled.append(idx)
    
    return subsampled

def create_unigram_table(word_freqs, word2idx, table_size=1e7):
    """Creates a table for negative sampling using freq^(3/4)."""
    sorted_vocab = sorted(word2idx.items(), key=lambda x: x[1])

    counts = np.array([word_freqs.get(word, 1e-6) for word, idx in sorted_vocab])
    
    pow_counts = counts**0.75
    probs = pow_counts / np.sum(pow_counts)
    
    return np.random.choice(len(word2idx), size=int(table_size), p=probs)
            
        
def get_target(words, idx, window_size=5):
    """
    Retrieves context words for a given center word.
    Uses dynamic window sizing.
    """
    R = np.random.randint(1, window_size + 1) # dynamic sizing
    
    start = max(0, idx - R)
    end = min(len(words), idx + R + 1)
    
    targets = words[start:idx] + words[idx+1:end]
    
    return targets

def get_batches(words, batch_size, window_size=5):
    """
    CORRECCIÓN: Ahora acumula pares hasta alcanzar exactamente el batch_size.
    Esto hace que los tensores sean uniformes y el entrenamiento más estable.
    """
    x, y = [], []
    for idx in range(len(words)):
        R = np.random.randint(1, window_size + 1)
        start = max(0, idx - R)
        end = min(len(words), idx + R + 1)
        
        center = words[idx]
        contexts = words[start:idx] + words[idx+1:end]
        
        x.extend([center] * len(contexts))
        y.extend(contexts)
        
        if len(x) >= batch_size:
            yield np.array(x[:batch_size]), np.array(y[:batch_size])
            x, y = x[batch_size:], y[batch_size:]
            
    if len(x) > 0:
        yield np.array(x), np.array(y)
    