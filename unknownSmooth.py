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

# Assigning the probabilities to one for addition
uniUnsmoothProb = 0
uniLaplaceProb = 0
uniAddKProb = 0
biUnsmoothProb = 0
biLaplaceProb = 0
biAddKProb = 0

with open('val.txt', 'r') as file:
    for line in file:
        # Split the first token and the rest of the line into a list of 2
        lineList = line.split(" ", 1)
        single = knownUni(lineList[0])
        
        # Number of tokens increase for the first word and end of sentence <s>
        numberOfTokens += 2
        
        # Muliplication of probabilities for the first token in the sentence
        uniLaplaceProb += uniLaplace(single)
        uniAddKProb += uniAddK(single, 0.01)
        uniUnsmoothProb += unigram_probs[single]
        
        for token in lineList[1].split():
            nextSingle = token
            nextSingle = knownUni(nextSingle)
            
            # Multiplcation of probabilities for the rest of the tokens in the sentence
            uniLaplaceProb += uniLaplace(nextSingle)
            uniAddKProb += uniAddK(nextSingle, 0.01)
            uniUnsmoothProb += unigram_probs[nextSingle]
            
            numberOfTokens += 1
            
            #Bigram token checked
            bigramWord = single + " " + nextSingle
            bigramWord = knownBi(bigramWord, single)
            biLaplaceProb += biLaplace(bigramWord, single)
            biAddKProb += biAddK(bigramWord, single, 0.01)
            biUnsmoothProb += bigram_probs[bigramWord]
            
            #For the next bigram
            single = nextSingle
