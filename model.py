import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        
        # W1 acts like the centre words matrix, W2 is the same for context.
        # We initialize with small random values to break symmetry
        self.W1 = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim)) / embedding_dim
        self.W2 = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim)) / embedding_dim
        
    def sigmoid(self, x):
        """Standard sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    
    def train_step(self, center_word_idx, context_word_idx, negative_indices):
        """
        """
        center_v = self.W1[center_word_idx]
        context_v = self.W2[context_word_idx]
        pos_pair = np.dot(center_v, context_v) # positive pair
        prob_pos = self.sigmoid(pos_pair)
        
        error_pos = prob_pos - 1 # error is defined as prediction - label, in this case label = 1
        
        negative_vs = self.W2[negative_indices] # multiple vectors
        neg_pairs = np.dot(center_v, negative_vs)
        probs_neg = self.sigmoid(neg_pairs)
        
        errors_neg = probs_neg # in this case label = 0
        
        gradient_center = error_pos * context_v + errors_neg * negative_vs
        
        grad_pos = error_pos * center_v
        grad_neg = np.outer(errors_neg, center_v) 
        
        self.W1[center_word_idx] -= self.lr * gradient_center
        self.W2[context_word_idx] -= self.lr * grad_pos
        self.W2[negative_indices] -= self.lr * grad_neg
        
        loss = -np.log(prob_pos + 1e-10) - np.sum(np.log(1 - probs_neg + 1e-10))
        return loss