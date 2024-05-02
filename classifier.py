import re
from collections import defaultdict
import csv
import math


ngram_select = ""

class UserTweet:
    def __init__(self, tweet_id, text, label):
        self.tweet_id = tweet_id
        self.text = text
        self.label = label


class User:
    def __init__(self, user_id, label):
        self.user_id = user_id
        self.tweets = []
        self.label = label

    def add_tweet(self, tweet):
        self.tweets.append(tweet)
    def __repr__(self) -> str:
        return str(self.user_id) + " " + str(self.label)


def clean_tweet(text):
    text = text.lower()

    # clean data of unnecessary symbols
    text = re.sub(r'#', ' ', text) # remove hashtags
    text = re.sub(r'@\w+', ' ', text) # remove @
    text = re.sub(r'http\S+', ' ', text) # remove links
    text = re.sub(r'[\d]+', '', text) #remove numbers
    text = re.sub(r"([\w])[']([\w])", r'\1\2', text)
    text = re.sub(r'[^\w\s]', ' ', text) # replacing punctation
    text = ' '.join(text.split()) # removing white space

    return text


#returns an array of tuples that looks like [ ("wordA", "wordB"), ("wordC", "wordD")]
def get_bigrams(text):
    words = text.split()
    bigrams = []
    for i in range(1, len(words)):
        bigrams.append((words[i - 1], words[i]))


    return bigrams

def get_unigrams(text):
    words = text.split()
    unigrams = []
    for word in words:
        unigrams.append((word,))
    
    return unigrams

def get_trigrams(text):
    words = text.split()
    trigrams = []
    for i in range(2, len(words)):
        trigrams.append((words[i - 2], words[i - 1], words[i]))

    return trigrams

def get_features(text):
    if ngram_select == "unigram":
        return get_unigrams(text)
    elif ngram_select == "bigram":
        return get_bigrams(text)
    elif ngram_select == "trigram":
        return get_trigrams(text)
    else:
        raise Exception("Need cli argument for n-gram type, choose from [unigram, bigram, trigram]")



def is_bad_feature(ngram):
    useless_words = ["i", "you", "and", "my", "it", "is", "that", "im", "the", "or", "of", "for", "zayn", "pillowtalk", "now", "gopaytwin", "yong", "paytforluckysun", "wearepayting", "foryong", "nowzayn", "bestmusicvideo", "iheartawards", "am", "a", "in", "your"]
    for word in ngram:
        if not word in useless_words:
            return False
    return True

#returns an array of user arrays that looks this [training_data, test_data]
def training_test_split(users, test_size=0.2):
    depressed_users = []
    non_depressed_users = []
    for user in users:
        if user.label == "1":
            depressed_users.append(user)
        else:
            non_depressed_users.append(user)

    training_data = []
    test_data = []
    num_users = len(users)
    training_size = 1 - test_size
    num_training_data = int(training_size * num_users)
    num_test_data = num_users - num_training_data

    training_depressed_prop = int(0.75 * num_training_data)
    training_non_depressed_prop = num_training_data - training_depressed_prop

    training_data = depressed_users[0:training_depressed_prop] + non_depressed_users[0:training_non_depressed_prop]
    test_data = depressed_users[training_depressed_prop:] + non_depressed_users[training_non_depressed_prop:]
    return [training_data, test_data]

def get_prior_prob(users):
    depressed_users = 0
    non_depressed_users = 0
    for user in users:
        if user.label == "1":
            depressed_users += 1
        else:
            non_depressed_users += 1
    
    return (depressed_users / len(users), non_depressed_users / len(users))


def get_conditional_prob(training_data, freq_bigrams):
    depressed_prob = {}
    non_depressed_prob = {}

    total_depressed = 0
    total_non_depressed = 0

    for user in training_data:
        current_bigrams = set()
        for tweet in user.tweets:
            current_bigrams.update(get_features(tweet.text))
        for current_bigram in current_bigrams:
            if current_bigram in freq_bigrams:
                if user.label == "1":
                    depressed_prob[current_bigram] = depressed_prob.get(current_bigram, 1) + 1
                    non_depressed_prob[current_bigram] = non_depressed_prob.get(current_bigram, 1)
                else:
                    depressed_prob[current_bigram] = depressed_prob.get(current_bigram, 1)
                    non_depressed_prob[current_bigram] = non_depressed_prob.get(current_bigram, 1) + 1
        
        if user.label == "1":
            total_depressed += 1
        else:
            total_non_depressed += 1

    for feature in depressed_prob:
        depressed_prob[feature] = depressed_prob[feature] / total_depressed
        non_depressed_prob[feature] = non_depressed_prob[feature] / total_non_depressed

    return [depressed_prob, non_depressed_prob]

def process_data(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        header = file.readline().strip().split(',')

        reader = csv.reader(file)

        users = []
        user_dict = {}
        
        for row in reader:
            # check to avoid index out of range error
            user_id = row[4]
            tweet_id = row[1]
            text = row[3]
            label = row[10]

            # if "yong" in text.lower():
            #     continue
            cleaned_text = clean_tweet(text)

            tweet = UserTweet(tweet_id, cleaned_text, label)

            # Check if user exists in the dict, else create a new User object
            if user_id not in user_dict:
                new_user = User(user_id, label)
                new_user.add_tweet(tweet)
                users.append(new_user)
                user_dict[user_id] = new_user
            else:
                user_dict[user_id].add_tweet(tweet)

    return users


# takes in float and dictionary (key: feature, val: probability of a class) and set of present features and returns class probability of a tweet
def naive_bayes(prior, feature_to_prob, present_features):
    product = 1
    for feature, val in feature_to_prob.items():
       # print('feature: ' + str(feature) + ' prob: '+ str(val))
        if feature in present_features:
            product *= val
        else:
            product *= 1 - val

    prob = prior * product
    return prob

def naive_bayes_log(prior, feature_to_prob, present_features):
    sum = math.log2(prior)
    for feature, val in feature_to_prob.items():
        if feature in present_features:
            sum += math.log2(val)
        else:
            sum += math.log2(1 - val)
    return sum

def classify_feature(user, depressed_dict, non_depressed_dict, prior_for_depressed, prior_for_non_depressed):
    bigrams = set()
    for tweet in user.tweets:
        bigrams.update(set(get_features(tweet.text))) # maybe needs fixing, needs to be looked at


    
    prob_depressed = naive_bayes_log(prior_for_depressed, depressed_dict, bigrams)
    prob_non_depressed = naive_bayes_log(prior_for_non_depressed, non_depressed_dict, bigrams)

    return "1" if prob_depressed > prob_non_depressed else "0"


def classify_tweet(tweet, depressed_dict, non_depressed_dict, prior_for_depressed, prior_for_non_depressed):
    bigrams = set()
    bigrams.update(set(get_features(tweet.text))) # maybe needs fixing, needs to be looked at

    prob_depressed = naive_bayes_log(prior_for_depressed, depressed_dict, bigrams)
    prob_non_depressed = naive_bayes_log(prior_for_non_depressed, non_depressed_dict, bigrams)

    return "1" if prob_depressed > prob_non_depressed else "0"
    return


def find_best_features( depressed_dict, non_depressed_dict, prior_depressed, prior_non_depressed):
    for feature in depressed_dict.keys():
        present_features = set()
        present_features.add(feature)
        prob_depressed = naive_bayes(prior_depressed, depressed_dict, present_features)
        prob_non_depressed = naive_bayes(prior_non_depressed, non_depressed_dict, present_features)



        depress_prob = (prob_depressed)/(prob_depressed + prob_non_depressed)


        if depress_prob >= .8:
            print(str(feature) + ' is good for depress')
        elif depress_prob <= .3:
            print(str(feature) + " is good for not")

def get_conditional_prob_by_tweet(training_tweets, freq_ngrams):
    depressed_prob = {}
    non_depressed_prob = {}

    total_depressed = 0
    total_non_depressed = 0

    for tweet in training_tweets:
        current_ngrams = set()
        current_ngrams.update(get_features(tweet.text))
        for current_ngram in current_ngrams:
            if current_ngram in freq_ngrams:
                if tweet.label == "1":
                    depressed_prob[current_ngram] = depressed_prob.get(current_ngram, 1) + 1
                    non_depressed_prob[current_ngram] = non_depressed_prob.get(current_ngram, 1)
                else:
                    depressed_prob[current_ngram] = depressed_prob.get(current_ngram, 1)
                    non_depressed_prob[current_ngram] = non_depressed_prob.get(current_ngram, 1) + 1
        
        if tweet.label == "1":
            total_depressed += 1
        else:
            total_non_depressed += 1

    for feature in depressed_prob:
        depressed_prob[feature] = depressed_prob[feature] / total_depressed
        non_depressed_prob[feature] = non_depressed_prob[feature] / total_non_depressed

    return [depressed_prob, non_depressed_prob]

def training_tweet_split(tweets, test_size):
    depressed_tweets = []
    non_depressed_tweets = []
    for tweet in tweets:
        if tweet.label == "1":
            depressed_tweets.append(tweet)
        else:
            non_depressed_tweets.append(tweet)

    training_data = []
    test_data = []
    num_tweets = len(tweets)
    training_size = 1 - test_size
    num_training_data = int(training_size * num_tweets)

    training_depressed_prop = int(0.5 * num_training_data)
    training_non_depressed_prop = num_training_data - training_depressed_prop

    training_data = depressed_tweets[0:training_depressed_prop] + non_depressed_tweets[0:training_non_depressed_prop]
    test_data = depressed_tweets[training_depressed_prop:] + non_depressed_tweets[training_non_depressed_prop:]
    return [training_data, test_data]

def run_classifier(users, training_data, test_data, minfreq, ngram_choice, log, print_features):
    global ngram_select
    ngram_select = ngram_choice
    words = set()


    ngrams = defaultdict(int)

    total_ngrams = 0
    total_words = 0

    depressed_count = 0

    depressed_tweets = 0
    non_depressed_tweets = 0


    for user in users:
        if user.label == "1":
            depressed_count += 1
            depressed_tweets += len(user.tweets)
        else:
            non_depressed_tweets += len(user.tweets)
        for tweet in user.tweets:
            words.update(tweet.text.split())
            total_words += len(tweet.text.split())
            ngram_list = get_features(tweet.text)
            total_ngrams += len(ngram_list)
            for ngram in ngram_list:
                ngrams[ngram] = ngrams[ngram] + 1

    training_tweets = []

    depressed_tweets = 0
    non_depressed_tweets = 0

    for user in training_data:
        training_tweets += user.tweets
        if user.label == "1":
            depressed_tweets += len(user.tweets)
        else:
            non_depressed_tweets += len(user.tweets)

    test_tweets = []

    for user in test_data:
        test_tweets += user.tweets

    for k, v in list(ngrams.items()):
        if v <= minfreq:
            del ngrams[k]
        elif is_bad_feature(k):
            del ngrams[k]

    if(print_features):
        print(ngrams)
            
    #depressed_dict, non_depressed_dict = get_conditional_prob(training_data, ngrams)

    depressed_tweet_dict, non_depressed_tweet_dict = get_conditional_prob_by_tweet(training_tweets, ngrams)






    #prior_for_tweets = (depressed_tweets/(depressed_tweets + non_depressed_tweets))

    prior_for_tweets = get_prior_prob(training_data)[0]
    correct = 0
    true_positives = 0
    false_positives = 0

    for user in test_data:
        if classify_feature(user, depressed_tweet_dict, non_depressed_tweet_dict, 0.5, 0.5) == user.label:
            correct += 1
            if user.label == "1":
                true_positives += 1
        if classify_feature(user, depressed_tweet_dict, non_depressed_tweet_dict, 0.5, 0.5) == "1":
            if user.label != "1":
                false_positives += 1
        elif log:
            classification = classify_feature(user, depressed_tweet_dict, non_depressed_tweet_dict, 0.5, 0.5)
            print('user id ' + str(user.user_id) + ' was classified as ' + str(classification) + ' but was actually ' + str(user.label))
    print('---------- ngram: ' + ngram_select + ' minfreq: ' + str(minfreq) + '--------------------')
    print('Test Data Accuracy: ' + str(correct) + ' users correct out of ' + str(len(test_data)))
    print('Test Data Precision: ' + str(true_positives) + ' users correct out of ' + str(true_positives + false_positives))

    correct = 0
    true_positives = 0
    false_positives = 0

    for user in training_data:
        if classify_feature(user, depressed_tweet_dict, non_depressed_tweet_dict, 0.5, 0.5) == user.label:
            correct += 1
            if user.label == "1":
                true_positives += 1
        if classify_feature(user, depressed_tweet_dict, non_depressed_tweet_dict, 0.5, 0.5) == "1":
            if user.label != "1":
                false_positives += 1
        elif log:
            classification = classify_feature(user, depressed_tweet_dict, non_depressed_tweet_dict, 0.5, 0.5)
            print('user id ' + str(user.user_id) + ' was classified as ' + str(classification) + ' but was actually ' + str(user.label))

    print('Training Data Accuracy: ' + str(correct) + ' users correct out of ' + str(len(training_data)))
    print('Training Data Precision: ' + str(true_positives) + ' users correct out of ' + str(true_positives + false_positives))
    print('--------------------------------------------------------------------------\n')
    if log:
        print(str(depressed_tweets) + ' total depressed tweets in training set')
        print(str(non_depressed_tweets) + ' total non_depressed tweets in training set')
        print('processed ' + str(len(ngrams)) + ' unique ngrams')
        print('processed ' + str(total_ngrams) + ' total ngrams')
        print('processed ' + str(total_words) + ' total words')
        print('processed ' + str(len(ngrams)) + ' frequent ngrams')
        print(str(depressed_count) + " depressed users")
        #print(bigrams)
