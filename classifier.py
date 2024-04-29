import re
from collections import defaultdict

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

    def __str__(self) -> str:
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

def get_bigrams(text):
    words = text.split()
    bigrams = []
    for i in range(1, len(words)):
        bigrams.append((words[i - 1], words[i]))

    return bigrams


def process_data(file_path):
    with open(file_path, 'r') as file:
        header = file.readline().strip().split(',')

        users = []
        user_dict = {}

        for line in file:
            row = line.strip().split(',')

            # check to avoid index out of range error
            if len(row) >= 11:
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

if log:
    print('processed ' + str(len(users)) + ' users')


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
if log:
    print('processed ' + str(len(words)) + ' unqique words')
    print('processed ' + str(len(bigrams)) + ' unique bigrams')
    print('processed ' + str(total_bigrams) + ' total bigrams')
    print('processed ' + str(total_words) + ' total words')
    for k, v in list(bigrams.items()):
        if v <= 15:
            del bigrams[k]
    print('processed ' + str(len(bigrams)) + ' frequent bigrams')
    print(str(depressed_count) + " depressed users")
    #print(bigrams)