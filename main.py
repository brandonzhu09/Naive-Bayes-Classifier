from classifier import *

file_path = 'Mental-Health-Twitter.csv'

log = False


users = process_data(file_path)

training_data, test_data = training_test_split(users, 0.2)



# FOR UNIGRAM DATA
# for i in range(0, 1000, 75):
#     run_classifier(users, training_data, test_data, i, "unigram", False, False)

#FOR BIGRAM DATA
for i in range(0, 350, 25):
    run_classifier(users, training_data, test_data, i, "bigram", False, False)
    

# for i in range (0, 50, 3):
#     run_classifier(users, training_data, test_data, i, "trigram", False, False)

# run_classifier(users, training_data, test_data, 295, "bigram", False, True)
