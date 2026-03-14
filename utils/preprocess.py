import os
import numpy as np
from collections import Counter
import random

# Define relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/wiki.train.tokens')

def load_text(file_path=DATA_PATH):
    """Reads the text file and returns a list of tokens."""
    
    print(f"Loading data from: {file_path}\n")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    
    with open(file_path, "r", "utf-8") as f:
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
    

def subsample_text(words, word2idx, word_freqs, threshold=1e-5):
    """
    Applies subsampling to frequent words based on Mikolov's formula.
    Returns the text converted to numerical indices.
    """
    subsampled = []
    
    for word in words:
        idx = word2idx.get(word, word2idx['<UNK>'])
        freq = word_freqs.get(word, 0)
        if freq > 0:
            p_keep = ((threshold/freq)**0.5) + (threshold/freq)
        else:
            p_keep = 1
    
        if random.random() < p_keep:
            subsampled.append(idx)
    
    return subsampled
            
        
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
    Generator that yields batches of (center_words, context_words).
    This is for training.
    """
    n_batches = len(words) // batch_size
    
    # We shorten the original list for a batch-sized list.
    words = words[:n_batches * batch_size]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_target(batch, i, window_size)
            
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
            
        yield np.array(x), np.array(y)
    