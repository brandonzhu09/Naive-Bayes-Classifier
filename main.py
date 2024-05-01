from classifier import *

file_path = 'Mental-Health-Twitter.csv'


users = process_data(file_path)


#users = [u for u in users if len(u.tweets) >= 100]

training_data, test_data = training_test_split(users, 0.2)


words = set()


bigrams = defaultdict(int)

total_bigrams = 0
total_words = 0

depressed_count = 0

depressed_tweets = 0
normal_tweets = 0

tweets = []

for user in users:
    tweets += user.tweets
    if user.label == "1":
        depressed_count += 1
        depressed_tweets += len(user.tweets)
    else:
        normal_tweets += len(user.tweets)
    for tweet in user.tweets:
        words.update(tweet.text.split())
        total_words += len(tweet.text.split())
        bigram_list = get_features(tweet.text)
        total_bigrams += len(bigram_list)
        for big in bigram_list:
            bigrams[big] = bigrams[big] + 1

training_tweets = []

depressed_tweets = 0
normal_tweets = 0

for user in training_data:
    training_tweets += user.tweets
    if user.label == "1":
        depressed_tweets += len(user.tweets)
    else:
        normal_tweets += len(user.tweets)

test_tweets = []

for user in test_data:
    test_tweets += user.tweets

for k, v in list(bigrams.items()):
    if v <= 50:
        del bigrams[k]
    elif is_bad_feature(k):
        del bigrams[k]

#print(bigrams)
        
depressed_dict, normal_dict = get_conditional_prob(training_data, bigrams)

depressed_tweet_dict, normal_tweet_dict = get_conditional_prob_by_tweet(training_tweets, bigrams)

#print(depressed_dict)
print("-------------------------------------------------------------------------------------------")
#print(normal_dict)

correct = 0


prior_for_depressed, prior_for_normal = get_prior_prob(training_data)


prior_for_tweets = (depressed_tweets/(depressed_tweets + normal_tweets))


find_best_features(depressed_tweet_dict, normal_tweet_dict, prior_for_tweets, 1 - prior_for_tweets)

for user in test_data:
    if classify_feature(user, depressed_tweet_dict, normal_tweet_dict, prior_for_tweets, 1 - prior_for_tweets) == user.label:
        correct += 1
    else:
        pass
        #print(user)
        #print(user.tweets[0].text)
        #print(classify_feature(user, depressed_dict, normal_dict, prior_for_depressed, prior_for_normal))

print('got ' + str(correct) + ' users correct out of ' + str(len(test_data)))

correct = 0

for user in training_data:
    if classify_feature(user, depressed_tweet_dict, normal_tweet_dict, prior_for_tweets, 1 - prior_for_tweets) == user.label:
        correct += 1
    else:
        pass
        #print(user)
        #print(user.tweets[0].text)
        #print(classify_feature(user, depressed_dict, normal_dict, prior_for_depressed, prior_for_normal))

print('got ' + str(correct) + ' users correct out of ' + str(len(training_data)))

correct = 0

for tweet in test_tweets:
    if classify_tweet(tweet, depressed_tweet_dict, normal_tweet_dict, prior_for_tweets, 1 - prior_for_tweets) == tweet.label:
        correct += 1

print('got ' + str(correct) + ' tweets correct out of ' + str(len(test_tweets)))


if log:
    print('processed ' + str(len(users)) + ' users')
    print(str(depressed_tweets) + ' total depressed tweets')
    print(str(normal_tweets) + ' total normal tweets')
    print('processed ' + str(len(words)) + ' unique words')
    print('processed ' + str(len(bigrams)) + ' unique bigrams')
    print('processed ' + str(total_bigrams) + ' total bigrams')
    print('processed ' + str(total_words) + ' total words')
    print('processed ' + str(len(bigrams)) + ' frequent bigrams')
    print(str(depressed_count) + " depressed users")
    #print(bigrams)
