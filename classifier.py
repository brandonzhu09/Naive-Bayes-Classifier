import re
from collections import defaultdict
import csv

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
    text = re.sub(r'#\w+', ' ', text) # remove hashtags
    text = re.sub(r'@\w+', ' ', text) # remove @
    text = re.sub(r'http\S+', ' ', text) # remove links
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

#returns an array of user arrays that looks this [training_data, test_data]
def training_test_split(users):
    depressed_users = []
    normal_users = []
    for user in users:
        if user.label == "1":
            depressed_users.append(user)
        else:
            normal_users.append(user)

    training_data = []
    test_data = []
    # training data - 57 users
    # test data - 15 users
    training_data = depressed_users[0:42] + normal_users[0:15]
    test_data = depressed_users[42:] + normal_users[15:]
    return [training_data, test_data]

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

    if log:
        print(len(user.tweets))

    for tweet in user.tweets:
        words.update(tweet.text.split())
        total_words += len(tweet.text.split())
        bigram_list = get_bigrams(tweet.text)
        total_bigrams += len(bigram_list)
        for big in bigram_list:
            bigrams[big] = bigrams[big] + 1


if log:
    print('processed ' + str(len(users)) + ' users')
    print('processed ' + str(len(words)) + ' unique words')
    print('processed ' + str(len(bigrams)) + ' unique bigrams')
    print('processed ' + str(total_bigrams) + ' total bigrams')
    print('processed ' + str(total_words) + ' total words')
    for k, v in list(bigrams.items()):
        if v <= 15:
            del bigrams[k]
    print('processed ' + str(len(bigrams)) + ' frequent bigrams')
    print(str(depressed_count) + " depressed users")
    #print(bigrams)


# takes in float and dictionary (key: feature, val: probability of a class) and set of present features and returns class probability of a tweet
def naive_bayes(prior, feature_to_prob, present_features):
    product = 1
    for feature, val in feature_to_prob.items():
        if feature in present_features:
            product *= val
        else:
            product *= 1 - val

    prob = prior * product
    return prob

def get_features(text):
    return get_bigrams(text)
def classify_feature(user, depressed_dict, normal_dict):
    bigrams = set()
    for tweet in user.tweets:
        bigrams.update(set(get_features(tweet.text)))


    prior_for_normal = 0 # replace
    prior_for_depressed = 0 # replace

    prob_depressed = naive_bayes(prior_for_depressed, depressed_dict, bigrams)
    prob_normal = naive_bayes(prior_for_normal, normal_dict, bigrams)

    return 1 if prob_depressed > prob_normal else 0
