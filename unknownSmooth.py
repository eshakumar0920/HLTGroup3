# Tentative name for unigram dictionary/hashmap: uni
# Tentative name for bigram dictionary/hashmap: bi
# Tentative name for unigram count dictionary/hashmap: uniCount
# Tentative name for bigram count dictionary/hashmap: biCount
# bigram token can have the two names combined to make it easier like "apple" and "tree" become "apple tree"
# If a dictionary is used, you can check unknown using word "in" dictionary. It will return false if the token is not in. 
# Change uniTotal into a float variable
# Add unknown token beforehand and set the value to 0

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
    if single in uni:
        return single
    else:
        uniCount["UNK"] += 1
        uniTotal += 1
        uni["UNK"] = uniCount["UNK"] / uniTotal
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
    if double in bi:
        return double
    else:
        biCount["UNK"] = biCount["UNK"] + 1
        bi["UNK"] = biCount[double] / float(uniCount[previous])
        return "UNK"

def uniLaplace(single):
    """
    The function calculates the Laplace smoothing for token with the unigram.
    
    Parameters:
        single (string): The token used for Laplace smoothing
    
    Returns:
        float: The probability based on Laplace smoothing for the unigram
    """
        return (uniCount[single] + 1) / (uniTotal + len(uni))

def biLaplace(double, previous):
    """
    The function calculates the Laplace smoothing for token with the bigram.
    
    Parameters:
        double (string): The token used for Laplace smoothing
        previous(string): The denominator for the token
    
    Returns:
        float: The probability based on Laplace smoothing for the bigram
    """
    return (biCount[double] + 1) / (float(uniCount[previous]) + len(bi))

def uniAddK(single, k):
    """
    The function calculates the Add-k smoothing for token with the unigram.
    
    Parameters:
        single (string): The token used for Add-k smoothing
    
    Returns:
        float: The probability based on Add-k smoothing for the unigram
    """
    return (uni[single] + k) / (uniTotal + (len(uni) * k))

def biAddK(double, previous, k):
    """
    The function calculates the Add-k smoothing for token with the bigram.
    
    Parameters:
        double (string): The token used for Add-k smoothing
        previous(string): The denominator for the token
    
    Returns:
        float: The probability based on Add-k smoothing for the bigram
    """
    return(bi[double] + k) / (float(uni[previous]) + (len(bi) * k))

numberOfTokens = 0

# Assigning the probabilities to one for multiplication
uniUnsmoothProb = 1
uniLaplaceProb = 1
uniAddKProb = 1
biUnsmoothProb = 1
biLaplaceProb = 1
biAddKProb = 1

with open('val.txt', 'r') as file:
    for line in file:
        # Split the first token and the rest of the line into a list of 2
        lineList = line.split(" ", 1)
        single = knownUni(lineList[0])
        
        # Number of tokens increase for the first word and end of sentence <s>
        numberOfTokens += 2
        
        # Muliplication of probabilities for the first token in the sentence
        uniLaplaceProb *= uniLaplace(single)
        uniAddKProb *= uniAddK(single, 0.01)
        uniUnsmoothProb *= uni[single]
        
        for token in lineList[1].split():
            nextSingle = token
            nextSingle = knownUni(nextSingle)
            
            # Multiplcation of probabilities for the rest of the tokens in the sentence
            uniLaplaceProb *= uniLaplace(nextSingle)
            uniAddKProb *= uniAddK(nextSingle, 0.01)
            uniUnsmoothProb *= uni[nextSingle]
            
            numberOfTokens += 1
            
            #Bigram token checked
            bigramWord = single + " " + nextSingle
            bigramWord = knownBi(bigramWord, single)
            biLaplaceProb *= biLaplace(bigramWord, single)
            biAddKProb *= biAddK(bigramWord, single, 0.01)
            biUnsmoothProb *= bi[bigramWord]
            
            #For the next bigram
            single = nextSingle
        
# The perplexity calculation
