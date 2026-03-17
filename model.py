import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embed_dim, lr):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = lr
        
        # CORRECCIÓN: W1 se inicializa uniforme pequeño, W2 EN CEROS.
        # Esto es clave para que el Skip-Gram arranque correctamente.
        self.W1 = np.random.uniform(-0.5/embed_dim, 0.5/embed_dim, (vocab_size, embed_dim))
        self.W2 = np.zeros((vocab_size, embed_dim))

    def _sigmoid(self, x):
        # Clip en -6, 6 (como en el paper de Mikolov) es suficiente
        return 1 / (1 + np.exp(-np.clip(x, -6, 6)))

    def train_step(self, centers, contexts, negatives):
        v_w = self.W1[centers]      
        v_c = self.W2[contexts]     
        v_n = self.W2[negatives]    

        # Forward
        pos_scores = np.sum(v_w * v_c, axis=1) 
        pos_probs = self._sigmoid(pos_scores)
        
        neg_scores = np.einsum('be,bne->bn', v_w, v_n)
        neg_probs = self._sigmoid(neg_scores)

        # Loss
        eps = 1e-10
        loss = -np.mean(np.log(pos_probs + eps) + np.sum(np.log(1 - neg_probs + eps), axis=1))

        # --- BACKPROPAGATION CORREGIDA PARA BATCHES ---
        batch_size = centers.shape[0]
        
        # ALERTA: Dividimos por batch_size para que el gradiente sea el promedio (igual que el np.mean del loss)
        grad_pos = ((pos_probs - 1) / batch_size).reshape(-1, 1) 
        grad_neg = (neg_probs / batch_size) 

        # 1. Gradiente para W2 (Contextos positivos)
        grad_W2_pos = grad_pos * v_w
        np.add.at(self.W2, contexts, -self.lr * grad_W2_pos)
        
        # 2. Gradiente para W2 (Negativos)
        grad_W2_neg = grad_neg[:, :, np.newaxis] * v_w[:, np.newaxis, :]
        np.add.at(self.W2, negatives, -self.lr * grad_W2_neg)

        # 3. Gradiente para W1 (Centro)
        grad_W1 = (grad_pos * v_c) + np.einsum('bn,bne->be', grad_neg, v_n)
        np.add.at(self.W1, centers, -self.lr * grad_W1)

        return loss