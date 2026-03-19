import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embed_dim, lr):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = lr
        
        """
        W1: Input/Center word embeddings. Initialized with small random values to 
        ensure numerical stability (avoiding sigmoid saturation/vanishing gradients) 
        and to break symmetry. If all vectors started equal, they would all 
        receive the same gradient updates and never learn distinct meanings.

        W2: Output/Context word embeddings. Initialized to zero so that the initial 
        dot product is 0, making the sigmoid return 0.5. This represents 
        maximum uncertainty (neutrality) and, crucially, provides the maximum 
        possible gradient to start the learning process.
        """
        self.W1 = np.random.uniform(-0.5/embed_dim, 0.5/embed_dim, (vocab_size, embed_dim))
        self.W2 = np.zeros((vocab_size, embed_dim))

    def _sigmoid(self, x):
        # Clipping to [-6, 6] ensures numerical stability and prevents np.exp from returning Inf or NaN.
        return 1 / (1 + np.exp(-np.clip(x, -6, 6)))

    def train_step(self, centers, contexts, negatives):
        v_w = self.W1[centers]      
        v_c = self.W2[contexts]     
        v_n = self.W2[negatives]    

        # Forward
        pos_scores = np.einsum('ij,ij->i', v_w, v_c)
        pos_probs = self._sigmoid(pos_scores)
        
        neg_scores = np.einsum('be,bne->bn', v_w, v_n)
        neg_probs = self._sigmoid(neg_scores)

        # Loss
        eps = 1e-10
        loss = -np.sum(np.log(pos_probs + eps) + np.sum(np.log(1 - neg_probs + eps), axis=1))
        
        # Backpropagation
        grad_pos = (pos_probs - 1).reshape(-1, 1) 
        grad_neg = neg_probs 
        
        """ 
        np.add.at ensures that if the same index appears multiple times in a batch 
        (frequent words), all its corresponding gradients are correctly 
        summed rather than overwritten.
        """

        # Gradient for W2 positive contexts
        grad_W2_pos = grad_pos * v_w
        np.add.at(self.W2, contexts, -self.lr * grad_W2_pos)
        
        # Gradient for W2 negative contexts
        grad_W2_neg = grad_neg[:, :, np.newaxis] * v_w[:, np.newaxis, :]
        np.add.at(self.W2, negatives, -self.lr * grad_W2_neg)

        # Gradient for W1
        grad_W1 = (grad_pos * v_c) + np.einsum('bn,bne->be', grad_neg, v_n)
        np.add.at(self.W1, centers, -self.lr * grad_W1)

        return loss