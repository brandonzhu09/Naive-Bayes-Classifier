from classifier import *

file_path = 'Mental-Health-Twitter.csv'

log = False


users = process_data(file_path)

training_data, test_data = training_test_split(users, 0.2)



#run_classifier(users, training_data, test_data, 550, "unigram", False, True)


# for i in range(0, 1000, 25):
#     run_classifier(users, training_data, test_data, i, "unigram", False, False)

# for i in range(0, 500, 25):
#     run_classifier(users, training_data, test_data, i, "bigram", False, False)
    

run_classifier(users, training_data, test_data, 52, "bigram", False, True)


# for i in range (0, 100, 1):
#     run_classifier(users, training_data, test_data, i, "trigram", False, False)

# run_classifier(users, training_data, test_data, 295, "bigram", False, True)
