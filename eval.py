import numpy as np
from utils.preprocessing import load_text, build_vocab

def get_similar_words(word_vec, W1, idx2word, top_k=5, exclude_words=None):
    """Encuentra palabras cercanas a un vector dado (no solo a una palabra)."""
    norms = np.linalg.norm(W1, axis=1)
    norms[norms == 0] = 1e-10 
    
    # Similitud de coseno entre el vector de entrada y toda la matriz W1
    similarity = np.dot(W1, word_vec) / (norms * np.linalg.norm(word_vec) + 1e-10)
    
    # Ordenar por similitud
    sorted_indices = np.argsort(similarity)[::-1]
    
    results = []
    for idx in sorted_indices:
        word = idx2word[idx]
        if exclude_words and word in exclude_words:
            continue
        results.append((word, similarity[idx]))
        if len(results) >= top_k:
            break
    return results

def solve_analogy(a, b, c, W1, word2idx, idx2word):
    """Resuelve la analogía: a es a b como c es a... (b - a + c)"""
    for w in [a, b, c]:
        if w not in word2idx:
            return f"Error: '{w}' no está en el vocabulario."

    # king - man + woman
    vec_a = W1[word2idx[a]]
    vec_b = W1[word2idx[b]]
    vec_c = W1[word2idx[c]]
    
    target_vec = vec_b - vec_a + vec_c
    
    print(f"\nResolviendo: {a} -> {b} como {c} -> ?")
    # Excluimos las palabras de la pregunta para no obtener "king" como respuesta a sí misma
    suggestions = get_similar_words(target_vec, W1, idx2word, exclude_words=[a, b, c])
    
    for word, score in suggestions:
        print(f"- {word} (Similitud: {score:.4f})")

if __name__ == "__main__":
    print("Cargando pesos y vocabulario...")
    raw_text = load_text()
    word2idx, idx2word, _ = build_vocab(raw_text)
    
    try:
        W1 = np.load('W1.npy')
        
        # Prueba de la analogía clásica
        # Nota: Asegúrate de que tus palabras en el corpus sean "man" o "men" según tu vocabulario
        solve_analogy("boy", "brother", "girl", W1, word2idx, idx2word)
        
        # Otra prueba: París es a Francia como Madrid es a...
        # solve_analogy("paris", "france", "madrid", W1, word2idx, idx2word)
        
    except FileNotFoundError:
        print("Error: 'W1.npy' no encontrado. Ejecuta el entrenamiento primero.")