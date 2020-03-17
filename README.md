## BurgerBot9000
Originally developed as a final project to MIT's 6.S191 (Introduction to Deep Learning) course, BurgerBot 9000 lives on as a memorial to the infinite data available on the internet, and a poorly-implemented AI's ability to interpret it. BurgerBot 9000 is a relatively simple machine, and it works by doing the following:

- Using the Twitter Search API to perform a full archive search for tweets containing the word 'burger'. 
- Using a Twitter-trained Sentiment Analysis tool (Sentiment140) to categorize tweets based on positive, netural, or negative sentiment.
- Using the postive tweets as training data for a LSTM-based Recurrent Neural Network, and training a model using TensorFlow.
- Using the trained model to generate new tweets, and posting the results on Twitter!

All of this culminates in the output you see here:
