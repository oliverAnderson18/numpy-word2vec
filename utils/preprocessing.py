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

def build_vocab(words, min_freq=20):
    """
    Builds vocabulary by filtering rare words.
    Returns mapping dictionaries and word frequencies.
    """
    print("Building vocabulary...\n")
    word_counts = Counter(words)
    
    vocab = [word for word, count in word_counts.items() if count >= min_freq]
    vocab = ["<UNK>"] + vocab  # If the filtered out word appears, we give it UNK.
    
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for i, word in enumerate(vocab)}
    
    total_words = sum(word_counts.values())
    word_freqs = {word: count / total_words for word, count in word_counts.items()}
    
    return word2idx, idx2word, word_freqs
    

def subsample_text(words, word2idx, word_freqs, threshold=1e-3):
    """
    Applies subsampling to frequent words based on Mikolov's subsampling formula.
    Returns the text converted to numerical indices.
    """
    subsampled = []

    for word in words:
        idx = word2idx.get(word, word2idx['<UNK>'])
        freq = word_freqs.get(word, threshold)

        if freq > threshold:
            p_keep = np.sqrt(threshold / freq) + (threshold / freq)
            p_keep = min(1.0, p_keep)
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
            

def get_batches(words, batch_size, window_size=5, buffer_size=500000):
    """
    Generates pairs for a batch, shuffles them, and yields them.
    This prevents freezing the RAM and allows training to start quickly.
    """
    for i in range(0, len(words), buffer_size):
        buffer_words = words[i : i + buffer_size + window_size] # A bit of overlap to not loose context
        pairs = []
        
        for idx in range(len(buffer_words)):
            # Dynamic windows
            R = np.random.randint(1, window_size + 1)
            start = max(0, idx - R)
            end = min(len(buffer_words), idx + R + 1)
            
            center = buffer_words[idx]
            # Extract the context ignoring the center
            contexts = [buffer_words[j] for j in range(start, end) if j != idx]
            
            for context in contexts:
                pairs.append((center, context))
        
        # Shuffle ONLY pairs of this buffer
        random.shuffle(pairs)
        
        for j in range(0, len(pairs), batch_size):
            batch = pairs[j : j + batch_size]
            if len(batch) < batch_size:
                continue
            
            x_batch = np.array([p[0] for p in batch])
            y_batch = np.array([p[1] for p in batch])
            yield x_batch, y_batch