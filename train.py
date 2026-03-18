import numpy as np
import random
from utils.preprocessing import (load_text, build_vocab, subsample_text,
                               create_unigram_table, get_batches)
from model import Word2Vec
from tqdm import tqdm

def train():
    
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    
    raw_text = load_text()
    word2idx, idx2word, word_freqs = build_vocab(raw_text)
    int_words = subsample_text(raw_text, word2idx, word_freqs)
    unigram_table = create_unigram_table(word_freqs, word2idx)

    # Hyperparameters
   # Hyperparameters sugeridos tras los cambios
    VOCAB_SIZE = len(word2idx)
    EMBEDDING_DIM = 100
    INITIAL_LR = 0.05  # <--- BAJADO de 1.0 a 0.05
    EPOCHS = 5
    BATCH_SIZE = 512
    NEG_SAMPLES = 5
    WINDOW_SIZE = 5    # Una ventana de 5 suele dar mejores resultados que 3

    model = Word2Vec(VOCAB_SIZE, EMBEDDING_DIM, INITIAL_LR)
    
    # Estimación de pares generados (cada palabra genera ~WINDOW_SIZE contextos)
    estimated_pairs = len(int_words) * WINDOW_SIZE
    total_steps = (estimated_pairs // BATCH_SIZE) * EPOCHS
    current_step = 0

    print(f"\nModel initialized: Vocab {VOCAB_SIZE}, Embeddings {EMBEDDING_DIM}")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        # ELIMINADO: shuffle(int_words) <- Esto destruía el contexto natural de las palabras
        
        # Al no tener una longitud fija de batches exacta, usamos tqdm genérico
        pbar = tqdm(enumerate(get_batches(int_words, BATCH_SIZE, window_size=WINDOW_SIZE)))
        
        for i, (x_batch, y_batch) in pbar:
            random_indices = np.random.randint(0, len(unigram_table), size=(len(x_batch), NEG_SAMPLES))
            neg_indices = unigram_table[random_indices]
            
            loss = model.train_step(x_batch, y_batch, neg_indices)
            
            # LR decay
            current_step += 1
            model.lr = max(INITIAL_LR * 0.0001, INITIAL_LR * (1.0 - current_step / total_steps))
            
            total_loss += loss
            
            if i % 50 == 0:
                pbar.set_description(f"Epoch {epoch+1} | Loss: {loss:.4f} | LR: {model.lr:.4f}")

        avg_epoch_loss = total_loss / max(1, i + 1)
        print(f"\n>>> Epoch {epoch+1} Finished. Avg Loss: {avg_epoch_loss:.4f} <<<")
        
    print("Training complete. Saving weights...")
    np.save('W1.npy', model.W1)
    np.save('W2.npy', model.W2)

if __name__ == "__main__":
    train()