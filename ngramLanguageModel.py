import re
from collections import Counter
import math  # for perplexity
from decimal import Decimal

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
    global total_tokens
    total_tokens = sum(unigram_counts.values())
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

# Smoothing

# Setting the unigram_counts, unigram_probs, bigram_counts, and bigram_probs to contain the "UNK".
unigram_counts["UNK"] = 0
unigram_probs["UNK"] = 0
bigram_counts["UNK UNK"] = 0
bigram_probs["UNK UNK"] = 0

def knownUni(single):
    """
    The function checks if the token given exists in the unigram dictionary;
    if it does, the token is given back.
    If it does not, it increases the unknown counter in the unigram counter and increases the total unigram counter both by one.
    Then it calculates and stores the probability for the unigram by dividing the unknown counter with the total unigram counter.
    Finally, the function returns the unknown token.
    
    Parameters:
        single (string): The token being checked
    
    Returns:
        string: The same token given or the unknown token
    """
    if single in unigram_probs:
        return single
    else:
        unigram_counts["UNK"] += 1
        global total_tokens
        total_tokens += 1
        unigram_probs["UNK"] = unigram_counts["UNK"] / total_tokens
        return "UNK"

def knownBi(double, previous):
    """
    The function checks if the token given exists in the bigram dictionary;
    if it does, the token is given back.
    If it does not, it increases the unknown counter in the bigram counter.
    Then it calculates and stores the probability for the bigram by
    dividing the unknown counter with the count of the previous token.
    Finally, the function returns the unknown token.
    
    Parameters:
        double (string): The token being checked
        previous (string): The denominator for the token
    
    Returns:
        string: The same token given or the unknown token
    """
    if double in bigram_probs:
        return double
    else:
        bigram_counts["UNK UNK"] += 1
        bigram_probs["UNK UNK"] = bigram_counts["UNK UNK"] / float(unigram_counts[previous])
        return "UNK UNK"

def uniLaplace(single):
    """
    The function calculates the Laplace smoothing for token with the unigram.
    
    Parameters:
        single (string): The token used for Laplace smoothing
    
    Returns:
        float: The probability based on Laplace smoothing for the unigram
    """
    return (unigram_counts[single] + 1) / (total_tokens + len(unigram_probs))

def biLaplace(double, previous):
    """
    The function calculates the Laplace smoothing for token with the bigram.
    
    Parameters:
        double (string): The token used for Laplace smoothing
        previous(string): The denominator for the token
    
    Returns:
        float: The probability based on Laplace smoothing for the bigram
    """
    return (bigram_counts[double] + 1) / (float(unigram_counts[previous]) + len(bigram_probs))

def uniAddK(single, k):
    """
    The function calculates the Add-k smoothing for token with the unigram.
    
    Parameters:
        single (string): The token used for Add-k smoothing
    
    Returns:
        float: The probability based on Add-k smoothing for the unigram
    """
    return (unigram_probs[single] + k) / (total_tokens + (len(unigram_probs) * k))

def biAddK(double, previous, k):
    """
    The function calculates the Add-k smoothing for token with the bigram.
    
    Parameters:
        double (string): The token used for Add-k smoothing
        previous(string): The denominator for the token
    
    Returns:
        float: The probability based on Add-k smoothing for the bigram
    """
    return(bigram_counts[double] + k) / (float(unigram_counts[previous]) + (len(bigram_probs) * k))

numberOfTokens = 0

# Assigning the probabilities
uniUnsmoothTotal = Decimal(0)
uniLaplaceTotal = 0
uniAddKTotal = 0
biUnsmoothTotal = 0
biLaplaceTotal = 0
biAddKTotal = 0

with open('val.txt', 'r') as file:
    for line in file:
        # Split the first token and the rest of the line into a list of 2
        lineList = line.split(" ", 1)
        single = knownUni(lineList[0])
        
        uniUnsmoothProb = 1
        uniLaplaceProb = 1
        uniAddKProb = 1
        biUnsmoothProb = 1
        biLaplaceProb = 1
        biAddKProb = 1
        
        # Number of tokens increase for the first word and end of sentence <s>
        numberOfTokens += 2
        
        # Muliplication of probabilities for the first token in the sentence
        uniLaplaceProb *= math.log(uniLaplace(single))
        uniAddKProb *= math.log(uniAddK(single, 0.01))
        uniUnsmoothProb *= math.log(unigram_probs[single])
        
        for token in lineList[1].split():
            nextSingle = token
            nextSingle = knownUni(nextSingle)
            
            # Multiplcation of probabilities for the rest of the tokens in the sentence
            uniLaplaceProb += math.log(uniLaplace(nextSingle))
            uniAddKProb += math.log(uniAddK(nextSingle, 0.01))
            uniUnsmoothProb += math.log(unigram_probs[nextSingle])
            
            numberOfTokens += 1
            
            #Bigram token checked
            bigramWord = single + " " + nextSingle
            bigramWord = knownBi(bigramWord, single)
            biLaplaceProb += math.log(biLaplace(bigramWord, single))
            biAddKProb += math.log(biAddK(bigramWord, single, 0.01))
            biUnsmoothProb += math.log(bigram_probs[bigramWord])
            
            #For the next bigram
            single = nextSingle

        """
        uniUnsmoothTotal += uniUnsmoothProb
        uniLaplaceTotal += uniLaplaceProb
        uniAddKTotal += uniAddKProb
        biUnsmoothTotal += biUnsmoothProb
        biLaplaceTotal += biLaplaceProb
        biAddKTotal += biAddKProb
        """
        
# The perplexity calculation
def split_line_into_ngrams(n, line, prev_n_tokens):
    ngrams = []
    tokens = prev_n_tokens + line  # track previous n-1 tokens from previous line

    # loop through each token in the line
    for i in range(len(tokens)-(n-1)):
        # create n-gram starting at token i
        ngram = tuple(tokens[i : i+n])
        # add ngram tuple to ngrams list
        ngrams.append(ngram)        
    
    return ngrams  

def accumulate_perplexity_values(n, tokens, prev_n_tokens):
    line_ngrams = split_line_into_ngrams(n, tokens, prev_n_tokens)
    #print("ngrams for this line: \n%s\n" % line_ngrams)
    num_tokens_in_line = len(tokens)
    #print(f"num_tokens_in_line: ", num_tokens_in_line)

    line_log_probability_sum = 0
    for ngram in line_ngrams:
        probability = 4   # calculate for unsmoothed, laplace, and addK --> make function that's called here, probUni(1-3) and probBi(1-3)
        line_log_probability_sum += math.log(probability)

    #print("line_log_probability_sum: %f\n" % line_log_probability_sum)
    return num_tokens_in_line, line_log_probability_sum

def get_perplexity_for_dataset(file_name, n):
    # N and log_probability_sum are variables to track the values needed to calculate the perplexity
    N = 0                       # total number of tokens in validation set
    log_probability_sum = 0     # total sum of each n-gram's probability in the validation set
    prev_n_tokens = []          # list of the last n-1 tokens from the previous line

    # tokenize the validation corpus, one line (review) at a time
    with open(file_name, "r") as dataset:
        for line in dataset:
            line_tokens = line.split()   # list of tokens in current line
            #print("Current Line:\n%s\n" % line_tokens)
            prev_n_tokens = line_tokens[-(n-1):]
            #print("Last n-1 tokens:\n%s\n" % prev_n_tokens)

            # Add current line's number of tokens and n-gram probabilities to N and log_probability_sum respectively
            N_temp, log_probability_sum_temp = accumulate_perplexity_values(n, line_tokens, prev_n_tokens)
            N += N_temp
            log_probability_sum += log_probability_sum_temp
            #print("UPDATED N, log_probability_sum: %s, %s\n" % (N, log_probability_sum))
        
    # After all lines in corpus are processed, get the perplexity for the validation set
    return math.exp(-log_probability_sum / N)

def test_perplexity(N, log_probability_sum):
    return math.exp(-log_probability_sum / N)

print(f"\nPerplexity for Unigrams")
print(f"Unsmoothed: ", test_perplexity(numberOfTokens, uniUnsmoothProb))
print(f"Laplace: ", test_perplexity(numberOfTokens, uniLaplaceProb))
print(f"AddK: ", test_perplexity(numberOfTokens, uniAddKProb))
print(f"\nPerplexity for Bigrams")
print(f"Unsmoothed: ", test_perplexity(numberOfTokens, biUnsmoothProb))
print(f"Laplace: ", test_perplexity(numberOfTokens, biLaplaceProb))
print(f"AddK: ", test_perplexity(numberOfTokens, biAddKProb))