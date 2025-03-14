import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
text=input("Enter the text: ")
sentiment = sia.polarity_scores(text)
print(sentiment)

if sentiment['compound'] > 0.05:
    print("Positive")
elif sentiment['compound'] < -0.05:
    print("Negative")
else:
    print("Neutral")
