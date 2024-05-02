from classifier import *

import classifier
import sys

#################################################

# TO RUN

# FIRST ARGUMENT IS NGRAM TYPE, CHOOSE FROM [unigram, bigram, trigram]

# SECOND ARGUMENT IS MINIMUM FREQUENCY, INPUT AN INTEGER >= 0
# THE SECOND ARGUMENT IS USED TO FILTER OUT FEATURES THAT DON'T APPEAR OFTEN ENOUGH IN THE
# DATASET, IF A FEATURE FREQUENCY <= MINIMUM FREQUENCY, 

#################################################

file_path = 'Mental-Health-Twitter.csv'

log = False


users = process_data(file_path)

training_data, test_data = training_test_split(users, 0.2)



#run_classifier(users, training_data, test_data, 550, "unigram", False, True)


for i in range(0, 1000, 25):
    run_classifier(users, training_data, test_data, i, "unigram", False, False)

# for i in range(0, 500, 25):
#     run_classifier(users, training_data, test_data, i, "bigram", False, False)
    
    
# for i in range (0, 100, 1):
#     run_classifier(users, training_data, test_data, i, "trigram", False, False)

# run_classifier(users, training_data, test_data, 295, "bigram", False, True)
