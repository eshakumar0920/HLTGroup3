import nltk     # Natural Language Toolkit
import numpy as np
import math


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

def get_perplexity_for_validation_data(file_name, n):
    # variables to track total N and total sum values
    N = 0                       # total number of tokens in validation set
    log_probability_sum = 0     # total sum of each n-gram's probability in the validation set
    prev_n_tokens = []

    # tokenize the validation corpus, one line (review) at a time
    with open(file_name, "r") as val_set:
        for line in val_set:
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
    return compute_perplexity(N, log_probability_sum)

def accumulate_perplexity_values(n, tokens, prev_n_tokens):
    line_ngrams = split_line_into_ngrams(n, tokens, prev_n_tokens)
    #print("ngrams for this line: \n%s\n" % line_ngrams)
    num_tokens_in_line = len(tokens)
    #print(f"num_tokens_in_line: ", num_tokens_in_line)

    line_log_probability_sum = 0
    for ngram in line_ngrams:
        probability = 4  # ?? need to calculate for this ngram
        line_log_probability_sum += math.log(probability)

    #print("line_log_probability_sum: %f\n" % line_log_probability_sum)
    return num_tokens_in_line, line_log_probability_sum

def compute_perplexity(N, log_probability_sum):
    return math.exp(-log_probability_sum / N)

if __name__ == "__main__":
    # Given DATASET contains "val.txt"  <-- validation set corpus!
    print("Perplexity: %.5f" % get_perplexity_for_validation_data("val.txt", 3))
