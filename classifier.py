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

def get_features(text):
    return get_bigrams(text)

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
    print('processed ' + str(len(bigrams)) + ' frequent bigrams')
    print(str(depressed_count) + " depressed users")
    #print(bigrams)
    print(get_prior_prob(users))