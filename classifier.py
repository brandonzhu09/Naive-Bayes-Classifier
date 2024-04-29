import re


class UserTweet:
    def __init__(self, tweet_id, text, label):
        self.tweet_id = tweet_id
        self.text = text
        self.label = label


class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.tweets = []

    def add_tweet(self, tweet):
        self.tweets.append(tweet)


def clean_tweet(text):
    text = text.lower()

    # clean data of unnecessary symbols
    text = re.sub(r'#\w+', ' ', text) # remove hashtags
    text = re.sub(r'@\w+', ' ', text) # remove @
    text = re.sub(r'http\S+', ' ', text) # remove links
    text = re.sub(r'[^\w\s]', ' ', text) # replacing punctation
    text = ' '.join(text.split()) # removing white space

    return text


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
                    new_user = User(user_id)
                    new_user.add_tweet(tweet)
                    users.append(new_user)
                    user_dict[user_id] = new_user
                else:
                    user_dict[user_id].add_tweet(tweet)

    return users


file_path = 'Mental-Health-Twitter.csv'


users = process_data(file_path)
for user in users:
    for tweet in user.tweets:
        print(tweet.text)

print(len(users))