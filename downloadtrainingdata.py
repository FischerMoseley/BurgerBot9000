import tweepy, sys, os, pickle, requests, json

# twitter API config
API_KEY_PATH = 'keys.json'

def configure_API(api_key_path):
    with open(api_key_path) as f:
        api_certs = json.load(f)
    
    auth = tweepy.AppAuthHandler(api_certs['key'], api_certs['secret'])
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    if (not api):
        print ("Can't Authenticate")
        sys.exit(-1)
    
    return api

def clean_tweet(tweet):
    cleaned_tweet = tweet.full_text
    if("RT @" in cleaned_tweet): return None

    # strip emoji and make the tweet ASCII characters only
    cleaned_tweet = cleaned_tweet.split('https://')[0].strip().encode('ascii', errors = 'ignore').decode('utf-8')

    # replace unruly characters will their ASCII equivalents
    cleaned_tweet = cleaned_tweet.replace('\n', ' ')
    cleaned_tweet = cleaned_tweet.replace('&amp', '&')
    cleaned_tweet = cleaned_tweet.replace('&gt', '>')
    cleaned_tweet = cleaned_tweet.replace('&lt', '<')
    cleaned_tweet = cleaned_tweet.replace('<;3', '')
    cleaned_tweet = ' '.join([substring for substring in cleaned_tweet.split() if '@' not in substring])


    # make sure that the tweet isn't a retweet and is still burger-related after all of our modifications
    # return the cleaned tweet if so
    if ("RT @" not in cleaned_tweet) and ('burger' in cleaned_tweet.lower()):
        return cleaned_tweet.lower()

def pull_tweets(api, searchQuery, maxTweets, tweetsPerQry, fName = 'training_data/all_tweets.txt'):
    # If results from a specific ID onwards are reqd, set since_id to that ID.
    # else default to no lower limit, go as far back as API allows
    sinceId = None

    # If results only below a specific ID are, set max_id to that ID.
    # else default to no upper limit, start from the most recent tweet matching the search query.
    max_id = 0

    output_list = []
    tweetCount = 0
    print("Downloading max {0} tweets".format(maxTweets))
    with open(fName, 'w') as f:
        while tweetCount < maxTweets:
            try:
                if (max_id <= 0):
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, tweet_mode='extended', lang = 'en')
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, tweet_mode='extended',
                                                since_id=sinceId, lang = 'en')
                else:
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, tweet_mode='extended', lang = 'en',
                                                max_id=str(max_id - 1))
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, tweet_mode='extended', lang = 'en',
                                                max_id=str(max_id - 1),
                                                since_id=sinceId)
                if not new_tweets:
                    print("No more tweets found")
                    break

                for tweet in new_tweets:
                    cleaned_tweet = clean_tweet(tweet)
                    if cleaned_tweet and cleaned_tweet not in output_list:
                        f.write(cleaned_tweet + '\n')
                        output_list.append(cleaned_tweet)

                tweetCount += len(new_tweets)
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id
                
            except tweepy.TweepError as e:
                # Just exit if any error
                print("some error : " + str(e))
                break

    print ("Downloaded {0} tweets, of which {1} were unique and were saved.".format(tweetCount, len(output_list)))
    return output_list

def get_tweet_sentiment(tweet_list, query):
    '''
    Utility function to classify sentiment of list of passed tweets using Sentiment140
    '''

    request_string = ''.join([ (tweet.encode('ascii', 'ignore').decode('utf-8') + '\n')for tweet in tweet_list])
    r = requests.post('http://www.sentiment140.com/api/bulkClassify', data = request_string, params = {'query':query} )
    
    if r.status_code != 200:
        raise requests.exceptions.RequestException()

    sentiment_list = r.content.decode("utf-8").strip().split('\n')
    sentiment_dict = {tweet[5:-1]:int(tweet[1]) for tweet in sentiment_list}
    return sentiment_dict

def save_tweets(tweet_dict):
    positive_tweet_list = [key for (key,value) in tweet_dict.items() if value == 4]
    neutral_tweet_list =  [key for (key,value) in tweet_dict.items() if value == 2]
    negative_tweet_list =  [key for (key,value) in tweet_dict.items() if value == 0]

    pickle.dump(positive_tweet_list, open("training_data/positive.pkl", "wb"))
    pickle.dump(neutral_tweet_list, open("training_data/neutral.pkl", "wb"))
    pickle.dump(negative_tweet_list, open("training_data/negative.pkl", "wb"))

    print("Found " + str(len(positive_tweet_list)) + " postitive tweets")
    print("Found " + str(len(neutral_tweet_list)) + " neutral tweets")
    print("Found " + str(len(negative_tweet_list)) + " negative tweets")

    with open("training_data/positive.txt", "w") as pos:
        pos.write("\n".join(positive_tweet_list))
    
    with open("training_data/neutral.txt", "w") as neut:
        neut.write("\n".join(neutral_tweet_list))
    
    with open("training_data/negative.txt", "w") as neg:
        neg.write("\n".join(negative_tweet_list))

searchQuery = 'burger'  # this is what we're searching for
maxTweets = 100 # Some arbitrary large number
tweetsPerQry = 100  # this is the max the API permits

api = configure_API(API_KEY_PATH)
output_list = pull_tweets(api, searchQuery, maxTweets, tweetsPerQry)
tweet_dict = get_tweet_sentiment(output_list, searchQuery)
save_tweets(tweet_dict)