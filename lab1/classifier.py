import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here 
    
    # Get number of words in SPAM bag and HAM bag
    spam_dict = util.get_word_freq(file_lists_by_category[0])
    ham_dict = util.get_word_freq(file_lists_by_category[1])
    
    num_words_spam_bag = sum(spam_dict.values())
    num_words_ham_bag = sum(ham_dict.values())
    
    # Get number of distinct words in the training set
    distinct_word_set = set()
    
    for word in spam_dict:
        distinct_word_set.add(word)
    for word in ham_dict:
        distinct_word_set.add(word)
        
    num_distinct_word = len(distinct_word_set)
    
    # Create Pd and Qd dictionaries
    Pd_dict = dict()
    Qd_dict = dict()
    
    for word in distinct_word_set:
        pd = 0
        
        # Words in the vocabulary and the SPAM bag
        if word in spam_dict:
            pd = (spam_dict[word] + 1) / (num_words_spam_bag + num_distinct_word)
        
        # Words in the vocabulary but not the SPAM bag
        else:
            pd = 1 / (num_words_spam_bag + num_distinct_word)
            
        Pd_dict[word] = pd
        
        qd = 0
        
        # Words in the vocabulary and the HAM bag
        if word in ham_dict:
            qd = (ham_dict[word] + 1) / (num_words_ham_bag + num_distinct_word)
        
        # Words in the vocabulary but not the HAM bag
        else:
            qd = 1 / (num_words_ham_bag + num_distinct_word)
            
        Qd_dict[word] = qd
    
    probabilities_by_category = (Pd_dict, Qd_dict)
    return probabilities_by_category

def classify_new_email(filename,probabilities_by_category,prior_by_category, k):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    
    # Get the frequency of words in the email (the X vector)
    words_freq_dict = util.get_word_freq([filename])
    
    # Calculate the log likelihood for both classes
    # If there are words not in the training vocabulary, ignore them (have a prob of 1 for both classes)
    log_spam_likelihood = 0
    log_ham_likelihood = 0
    
    for word in words_freq_dict:
        if word in probabilities_by_category[0]:
            log_spam_likelihood += np.log(probabilities_by_category[0][word]) * words_freq_dict[word]
        if word in probabilities_by_category[1]:
            log_ham_likelihood += np.log(probabilities_by_category[1][word]) * words_freq_dict[word]

    
    # Apply Map Rule
    if log_spam_likelihood - log_ham_likelihood >= np.log(k):
        y_hat = 'spam'
    else:
        y_hat = 'ham'

    classify_result = (y_hat, (log_spam_likelihood, log_ham_likelihood))
    
    return classify_result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category, 1)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off 
    
    # Initialize two arrays for x-values and y-values
    type_1_error = []
    type_2_error = []
    
    ratio_between_priors = [1e-4, 1e-2, 1, 5, 1e5, 1e8, 1e10, 1e14, 1e17, 1e23]
    
    for k in ratio_between_priors:
        
        performance_measures2 = np.zeros([2,2])
        
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename,
                                                     probabilities_by_category,
                                                     priors_by_category, k)
            
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            performance_measures2[int(true_index), int(guessed_index)] += 1

        # Correct counts are on the diagonal
        correct2 = np.diag(performance_measures2)
        # totals are obtained by summing across guessed labels
        totals2 = np.sum(performance_measures2, 1)
        
        # Get the error
        type_1_error.append(totals2[0] - correct2[0])
        type_2_error.append(totals2[1] - correct2[1])
    
    # Trade off plot
    plt.xlabel("# of Type 1 Errors")
    plt.ylabel("# of Type 2 Errors")
    plt.title("Trade Off between Type 1 Error and Type 2 Errors")
    plt.plot(type_1_error, type_2_error)
    plt.show()
        

 