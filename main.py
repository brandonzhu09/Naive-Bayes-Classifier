from classifier import *

file_path = 'Mental-Health-Twitter.csv'


users = process_data(file_path)
training_data, test_data = training_test_split(users)


words = set()


bigrams = defaultdict(int)

total_bigrams = 0
total_words = 0

depressed_count = 0

for user in users:
    if user.label == "1":
        depressed_count += 1

    for tweet in user.tweets:
        words.update(tweet.text.split())
        total_words += len(tweet.text.split())
        bigram_list = get_bigrams(tweet.text)
        total_bigrams += len(bigram_list)
        for big in bigram_list:
            bigrams[big] = bigrams[big] + 1


for k, v in list(bigrams.items()):
    if v <= 25:
        del bigrams[k]
        
get_conditional_prob(training_data, bigrams)


if log:
    print('processed ' + str(len(users)) + ' users')
    print('processed ' + str(len(words)) + ' unique words')
    print('processed ' + str(len(bigrams)) + ' unique bigrams')
    print('processed ' + str(total_bigrams) + ' total bigrams')
    print('processed ' + str(total_words) + ' total words')
    for k, v in list(bigrams.items()):
        if v <= 25:
            del bigrams[k]
    print('processed ' + str(len(bigrams)) + ' frequent bigrams')
    print(str(depressed_count) + " depressed users")
    #print(bigrams)
