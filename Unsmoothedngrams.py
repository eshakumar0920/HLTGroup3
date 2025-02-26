import re
from collections import Counter

total_tokens = 0

def preprocess(text):
    """Lowercase and tokenize text with improved normalization."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", "", text)  # Allow contractions like "don't"
    tokens = text.split()
    return tokens

def compute_unigram_probabilities(tokens):
    """Compute unigram probabilities."""
    unigram_counts = Counter(tokens)
    global total_tokens = sum(unigram_counts.values())
    unigram_probs = {word: count / total_tokens for word, count in unigram_counts.items()}
    return unigram_probs, unigram_counts

def compute_bigram_probabilities(tokens, min_count=2):
    """Compute bigram probabilities, filtering out rare bigrams."""
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(tokens)
    
    # Filter out rare bigrams
    bigram_counts = {bigram: count for bigram, count in bigram_counts.items() if count >= min_count}
    
    bigram_probs = {bigram: count / unigram_counts[bigram[0]] for bigram, count in bigram_counts.items()}
    return bigram_probs, bigram_counts

def process_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokens = preprocess(text)
    unigram_probs, unigram_counts = compute_unigram_probabilities(tokens)
    bigram_probs, bigram_counts = compute_bigram_probabilities(tokens)
    
    return unigram_probs, bigram_probs, unigram_counts, bigram_counts

if __name__ == "__main__":
    file_path = "train.txt"  # Adjust path if necessary
    unigram_probs, bigram_probs, unigram_counts, bigram_counts = process_corpus(file_path)
    
    print("Sample Unigram Probabilities:")
    for word, prob in list(unigram_probs.items())[:10]:  # Display first 10 unigrams
        print(f"P({word}) = {prob:.4f}")
    
    print("\nSample Bigram Probabilities:")
    for bigram, prob in list(bigram_probs.items())[:10]:  # Display first 10 bigrams
        print(f"P({bigram[1]} | {bigram[0]}) = {prob:.4f}")
