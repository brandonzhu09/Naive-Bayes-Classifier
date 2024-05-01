import re
from collections import defaultdict
import csv
import math

log = True

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
        unigrams.append((word))
    
    return unigrams

def get_trigrams(text):
    words = text.split()
    trigrams = []
    for i in range(2, len(words)):
        trigrams.append((words[i - 2], words[i - 1], words[i]))

    return trigrams

def get_features(text):
    return get_bigrams(text)


useless_words = ["i", "you", "and", "my", "it", "is", "that", "im", "the", "or", "of", "for", "zayn", "pillowtalk", "now", "gopaytwin", "yong", "paytforluckysun", "wearepayting", "nowzayn", "bestmusicvideo", "iheartawards", "am", "a", "in", "your"]

def is_bad_feature(ngram):
    for word in ngram:
        if word not in useless_words:
            return False
    return True

#returns an array of user arrays that looks this [training_data, test_data]
def training_test_split(users, test_size=0.2):
    depressed_users = []
    normal_users = []
    for user in users:
        if user.label == "1":
            depressed_users.append(user)
        else:
            normal_users.append(user)

    training_data = []
    test_data = []
    num_users = len(users)
    training_size = 1 - test_size
    num_training_data = int(training_size * num_users)
    num_test_data = num_users - num_training_data

    training_depressed_prop = int(0.75 * num_training_data)
    training_normal_prop = num_training_data - training_depressed_prop

    print(training_depressed_prop, training_normal_prop)

    training_data = depressed_users[0:training_depressed_prop] + normal_users[0:training_normal_prop]
    test_data = depressed_users[training_depressed_prop:] + normal_users[training_normal_prop:]
    return [training_data, test_data]

def get_prior_prob(users):
    depressed_users = 0
    normal_users = 0
    for user in users:
        if user.label == "1":
            depressed_users += 1
        else:
            normal_users += 1
    
    return (depressed_users / len(users), normal_users / len(users))


def get_conditional_prob(training_data, freq_bigrams):
    depressed_prob = {}
    normal_prob = {}

    total_depressed = 0
    total_normal = 0

    for user in training_data:
        current_bigrams = set()
        for tweet in user.tweets:
            current_bigrams.update(get_features(tweet.text))
        for current_bigram in current_bigrams:
            if current_bigram in freq_bigrams:
                if user.label == "1":
                    depressed_prob[current_bigram] = depressed_prob.get(current_bigram, 1) + 1
                    normal_prob[current_bigram] = normal_prob.get(current_bigram, 1)
                else:
                    depressed_prob[current_bigram] = depressed_prob.get(current_bigram, 1)
                    normal_prob[current_bigram] = normal_prob.get(current_bigram, 1) + 1
        
        if user.label == "1":
            total_depressed += 1
        else:
            total_normal += 1

    for feature in depressed_prob:
        depressed_prob[feature] = depressed_prob[feature] / total_depressed
        normal_prob[feature] = normal_prob[feature] / total_normal

    return [depressed_prob, normal_prob]

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

def classify_feature(user, depressed_dict, normal_dict, prior_for_depressed, prior_for_normal):
    bigrams = set()
    for tweet in user.tweets:
        bigrams.update(set(get_features(tweet.text))) # maybe needs fixing, needs to be looked at


    
    prob_depressed = naive_bayes_log(prior_for_depressed, depressed_dict, bigrams)
    prob_normal = naive_bayes_log(prior_for_normal, normal_dict, bigrams)

    return "1" if prob_depressed > prob_normal else "0"


def classify_tweet(tweet, depressed_dict, normal_dict, prior_for_depressed, prior_for_normal):
    bigrams = set()
    bigrams.update(set(get_features(tweet.text))) # maybe needs fixing, needs to be looked at

    prob_depressed = naive_bayes_log(prior_for_depressed, depressed_dict, bigrams)
    prob_normal = naive_bayes_log(prior_for_normal, normal_dict, bigrams)

    return "1" if prob_depressed > prob_normal else "0"
    return


def find_best_features( depressed_dict, normal_dict, prior_depressed, prior_normal):
    for feature in depressed_dict.keys():
        present_features = set()
        present_features.add(feature)
        prob_depressed = naive_bayes(prior_depressed, depressed_dict, present_features)
        prob_normal = naive_bayes(prior_normal, normal_dict, present_features)



        depress_prob = (prob_depressed)/(prob_depressed + prob_normal)


        if depress_prob >= .8:
            print(str(feature) + ' is good for depress')
        elif depress_prob <= .3:
            print(str(feature) + " is good for not")

def get_conditional_prob_by_tweet(training_tweets, freq_bigrams):
    depressed_prob = {}
    normal_prob = {}

    total_depressed = 0
    total_normal = 0

    for tweet in training_tweets:
        current_bigrams = set()
        current_bigrams.update(get_features(tweet.text))
        for current_bigram in current_bigrams:
            if current_bigram in freq_bigrams:
                if tweet.label == "1":
                    depressed_prob[current_bigram] = depressed_prob.get(current_bigram, 1) + 1
                    normal_prob[current_bigram] = normal_prob.get(current_bigram, 1)
                else:
                    depressed_prob[current_bigram] = depressed_prob.get(current_bigram, 1)
                    normal_prob[current_bigram] = normal_prob.get(current_bigram, 1) + 1
        
        if tweet.label == "1":
            total_depressed += 1
        else:
            total_normal += 1


    print("total depressed tweets read is " + str(total_depressed))
    print("total normal tweets read is " + str(total_normal))

    for feature in depressed_prob:
        depressed_prob[feature] = depressed_prob[feature] / total_depressed
        normal_prob[feature] = normal_prob[feature] / total_normal

    return [depressed_prob, normal_prob]

def training_tweet_split(tweets, test_size):
    depressed_tweets = []
    normal_tweets = []
    for tweet in tweets:
        if tweet.label == "1":
            depressed_tweets.append(tweet)
        else:
            normal_tweets.append(tweet)

    training_data = []
    test_data = []
    num_tweets = len(tweets)
    training_size = 1 - test_size
    num_training_data = int(training_size * num_tweets)

    training_depressed_prop = int(0.5 * num_training_data)
    training_normal_prop = num_training_data - training_depressed_prop

    print(training_depressed_prop, training_normal_prop)

    training_data = depressed_tweets[0:training_depressed_prop] + normal_tweets[0:training_normal_prop]
    test_data = depressed_tweets[training_depressed_prop:] + normal_tweets[training_normal_prop:]
    return [training_data, test_data]