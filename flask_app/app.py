import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
import string

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Upload the chat file
    file = request.files['chat_file']
    data = []
    conversation = file.read().decode('utf-8')
    lines = conversation.split('\n')
    
    def date_time(s):
        pattern = '^([0-9]+(\/)([0-9]+)(\/)[0-9]+, ([0-9]+):([0-9]+)\s(PM|AM|am|pm) - )'
        result = re.match(pattern, s)
        if result:
            return True 
        return False

    def contact(s):
        s = s.split(":")
        if len(s) == 2:
            return True 
        return False

    def getmsg(line):
        splitline = line.split(' - ')
        date, time = splitline[0].split(', ')
        msg = " ".join(splitline[1:])
        if contact(msg):
            split_msg = msg.split(': ')
            author = split_msg[0]
            msg = " ".join(split_msg[1:])
        else:
            author = None
        return date, time, author, msg

    msgBuffer = []
    date, time, author = None, None, None
    for line in lines:
        line = line.strip()
        if date_time(line):
            if len(msgBuffer) > 0:
                data.append([date, time, author, " ".join(msgBuffer)])
            msgBuffer.clear()
            date, time, author, msg = getmsg(line)
            msgBuffer.append(msg)
        else:
            msgBuffer.append(line)
    if len(msgBuffer) > 0:
        data.append([date, time, author, " ".join(msgBuffer)])
    
    data = pd.DataFrame(data, columns=["Date", "Time", "Contact", "Message"])
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.dropna()

    # Cleaning the Data
    stop_words = set(stopwords.words('english'))

    # Cleaning the deleted messages and media omitted logs
    data["Message"] = data['Message'][data["Message"] != "<Media omitted>"]
    data.dropna(axis=0, inplace=True)
    string_to_match = "deleted this message"
    data = data[~data["Message"].str.contains(string_to_match, case=False)]

    # Cleaning the messages from punctuations and stopwords and tokenized the messages
    data['Cleaned_message'] = data["Message"].apply(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)))
    data['Tokenized_words'] = data["Cleaned_message"].apply(lambda y: [word for word in word_tokenize(y) if word not in stop_words])

    # Applying the lemmatization techniques
    lemmatizer = WordNetLemmatizer()
    data["Lemmatized"] = data["Tokenized_words"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # Sentiment analysis
    sentiments = SentimentIntensityAnalyzer()
    data['Positive'] = data['Message'].apply(lambda x: sentiments.polarity_scores(x)['pos'])
    data['Negative'] = data['Message'].apply(lambda x: sentiments.polarity_scores(x)['neg'])
    data['Neutral'] = data['Message'].apply(lambda x: sentiments.polarity_scores(x)['neu'])
    
    data["Positive"] = data["Positive"].apply(lambda x: np.ceil(x) if x - np.floor(x) >= 0.5 else np.floor(x))
    data["Negative"] = data["Negative"].apply(lambda x: np.ceil(x) if x - np.floor(x) >= 0.5 else np.floor(x))
    data["Neutral"] = data["Neutral"].apply(lambda x: np.ceil(x) if x - np.floor(x) >= 0.5 else np.floor(x))

    def compare_values(row):
        if row['Positive'] > row['Negative'] and row['Positive'] > row['Neutral']:
            return 'positive'
        elif row['Negative'] > row['Positive'] and row['Negative'] > row['Neutral']:
            return 'negative'
        else:
            return 'neutral'
    data["sentiment"] = data.apply(compare_values, axis=1)

    # Extracting the sentiment trends into separate dataframe
    grouped = data.groupby('Date')
    def most_common_sentiment(series):
        return Counter(series).most_common(1)[0][0]

    sentiment_logs = grouped.agg(
        start_sentiment=('sentiment', 'first'),
        end_sentiment=('sentiment', 'last'),
        most_common_sentiment=('sentiment', most_common_sentiment)
    ).reset_index()

    # Add columns for the first and last Sentiment values within each group
    data['Start Conversation'] = grouped['sentiment'].transform('first')
    data['Stop Conversation'] = grouped['sentiment'].transform('last')

    # Analyzing the word frequencies
    combined_words = [word for sublist in data['Lemmatized'] for word in sublist]
    fdist = nltk.FreqDist(combined_words)
    common_words = fdist.most_common(10)

    # Topic modeling
    data["Tokenized_mgs"] = data["Tokenized_words"].apply(lambda x: " ".join(x))

    # LDA
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['Tokenized_mgs'])
    lda_model = LatentDirichletAllocation(n_components=5, random_state=0)
    lda_model.fit(X)
    topic_distribution = lda_model.transform(X)
    def most_relevant(topic_distribution):
        most_relevant_topics = []
        for distribution in topic_distribution:
            most_relevant_topic = distribution.argmax()
            most_relevant_topics.append(most_relevant_topic)
        return most_relevant_topics
    relevant_topics = most_relevant(topic_distribution)
    data['LDA'] = relevant_topics

    print(fdist)
    # Visualization for word cloud and message volumes
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(fdist)
    wordcloud.to_file('./static/images/wordcloud.png')
    
    def get_sentiment(text):
        sentiment = sentiments.polarity_scores(text)
        return sentiment['compound']

    data['sentiment_compound'] = data['Tokenized_mgs'].apply(get_sentiment)

    # Aggregate sentiment scores by date
    data["Date"] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.date
    daily_sentiment = data.groupby('Date')['sentiment_compound'].mean().reset_index()

    # Plot sentiment trends
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=daily_sentiment, x='Date', y='sentiment_compound', marker='o')
    plt.title('Sentiment Trend')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment')
    plt.grid(True)
    plt.savefig('static/images/sentiment_trend.png')
    plt.close()

    message_counts = data.groupby('Date').size().reset_index(name='Message Count')
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=message_counts, x='Date', y='Message Count', marker='o')
    plt.title('Message Volumes Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.grid(True)
    plt.savefig('static/images/message_volumes.png')
    plt.close()

    # Send results to the front end
    return render_template('results.html', common_words=common_words, sentiment_logs=sentiment_logs.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
