import numpy as np
from utils.preprocessing import load_text, build_vocab

def get_similar_words(word_vec, W1, idx2word, top_k=5, exclude_words=None):
    """Finds the closests words to the given words."""
    norms = np.linalg.norm(W1, axis=1)
    norms[norms == 0] = 1e-10 
    
    # Cosine similarity between W1 and the input vector
    similarity = np.dot(W1, word_vec) / (norms * np.linalg.norm(word_vec) + 1e-10)
    
    # Order by similarity
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


if __name__ == "__main__":
    print("Loading weights and vocabulary...")
    raw_text = load_text()
    word2idx, idx2word, _ = build_vocab(raw_text)
    
    try:
        W1 = np.load('W1.npy')
        
        palabras_objetivo = ["king", "football", "apple", "car", "foot"]
        
        print("\n=== Searching for the 5 closest words ===")
        for palabra in palabras_objetivo:
            # We must verify that the word exists
            if palabra not in word2idx:
                print(f"\n[!] The word '{palabra}' is not in the vocab")
                continue
            
            # Obtain the embedding of the word
            vec_palabra = W1[word2idx[palabra]]
            
            # Find the closest words (we must exclude itself)
            similares = get_similar_words(vec_palabra, W1, idx2word, top_k=5, exclude_words=[palabra])
            
            print(f"\nClosest words to '{palabra}':")
            for sim_word, score in similares:
                print(f"  * {sim_word} (Similarity: {score:.4f})")
                
    except FileNotFoundError:
        print("Error: 'W1.npy' not found. Execute train.py first.")