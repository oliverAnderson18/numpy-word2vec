import numpy as np
from utils.preprocessing import load_text, build_vocab

def get_similar_words(word_vec, W1, idx2word, top_k=5, exclude_words=None):
    """Finds the closest words to the given word vector."""
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
        
        print("\n" + "="*45)
        print("   Word Similarity Search Tool")
        print("   (Enter '0' to exit the program)")
        print("="*45)

        while True:
            # Get user input
            target_word = input("\nEnter a word: ").strip().lower()

            # Exit condition
            if target_word == '0':
                print("Exiting... Goodbye!")
                break

            # Check if the word exists in the vocabulary
            if target_word not in word2idx:
                print(f"[!] The word '{target_word}' was not found in the vocabulary.")
                continue
            
            # Obtain the word embedding 
            word_vector = W1[word2idx[target_word]]
            
            # Find the closest words
            similar_results = get_similar_words(word_vector, W1, idx2word, top_k=5, exclude_words=[target_word])
            
            print(f"\nTop matches for '{target_word}':")
            for word, score in similar_results:
                print(f"  * {word:<15} (Similarity: {score:.4f})")
                
    except FileNotFoundError:
        print("Error: 'W1.npy' not found. Please run train.py first to generate the weights.")