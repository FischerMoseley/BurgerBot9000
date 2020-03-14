import tweepy, sys, os, pickle, requests, json

# twitter API config
API_KEY_PATH = 'keys.json'

def configureAPI(api_key_path):
    with open(api_key_path) as f:
        api_certs = json.load(f)
    
    auth = tweepy.AppAuthHandler(api_certs['key'], api_certs['secret'])
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    if (not api):
        print ("Can't Authenticate")
        sys.exit(-1)
    
    return api

def pullTweets(api, searchQuery, maxTweets, tweetsPerQry):
    # If results from a specific ID onwards are reqd, set since_id to that ID.
    # else default to no lower limit, go as far back as API allows
    sinceId = None

    # If results only below a specific ID are, set max_id to that ID.
    # else default to no upper limit, start from the most recent tweet matching the search query.
    max_id = -1

    output_list = []
    tweetCount = 0
    print("Downloading max {0} tweets".format(maxTweets))

    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, tweet_mode='extended', lang = 'en')
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, since_id=sinceId, tweet_mode='extended', lang = 'en')
            else:
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, max_id=str(max_id - 1), tweet_mode='extended', lang = 'en')
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, max_id=str(max_id - 1), since_id=sinceId, tweet_mode='extended', lang = 'en')

            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                if ('RT @' not in tweet.full_text): #don't include retweets in the corpus
                    cleaned_tweet = tweet.full_text.split('https://')[0].strip()
                    output_list.append(cleaned_tweet)

            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
            
        except tweepy.TweepError as e: # Just exit if any error
            print("some error : " + str(e))
            break

    print ("Downloaded {0} tweets, of which {1} were not retweets and were saved.".format(tweetCount, len(output_list)))
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
    sentiment_dict = {tweet[4:-2]:int(tweet[1]) for tweet in sentiment_list}
    return sentiment_dict

def save_tweets(tweet_dict):
    positive_tweet_dict = 
    neutral_tweet_dict = 
    negative_tweet_dict = 

    pickle.dump(positive_tweet_dict, open("positive.pkl", "wb"))
    pickle.dump(neutral_tweet_dict, open("neutral.pkl", "wb"))
    pickle.dump(negative_tweet_dict, open("negative.pkl", "wb"))

    print("Found " + str(len(positive_tweet_dict)) + " postitive tweets")
    print("Found " + str(len(neutral_tweet_dict)) + " neutral tweets")
    print("Found " + str(len(negative_tweet_dict)) + " negative tweets")


searchQuery = 'burger'  # this is what we're searching for
maxTweets = 10000 # Some arbitrary large number
tweetsPerQry = 100  # this is the max the API permits
output_file = 'tweets.txt' # We'll store the tweets in a text file.

api = configureAPI(API_KEY_PATH)
output_list = pullTweets(api, searchQuery, maxTweets, tweetsPerQry)
tweet_dict = get_tweet_sentiment(output_list, searchQuery)

