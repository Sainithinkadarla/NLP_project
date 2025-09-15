import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
import string
import io

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet


# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = FastAPI()

# Helper functions for data parsing (same as your original code)
def date_time(s):
    pattern = r'^([0-9]+(\/)[0-9]+(\/)[0-9]+, ([0-9]+):([0-9]+)\s(PM|AM|am|pm) - )'
    return re.match(pattern, s) is not None

def contact(s):
    return len(s.split(":")) == 2

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

def create_pdf_report_reportlab(common_words, sentiment_logs, images_data):
    # Use an in-memory buffer
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title= "Chat Analysis Report")
    story = []
    styles = getSampleStyleSheet()

    # Title
    story.append(Paragraph("<b>Chat Analysis Report</b>", styles['Title']))
    story.append(Spacer(1, 12))

    # Most Common Words
    story.append(Paragraph("<b>Most Common Words</b>", styles['Heading2']))
    for word, count in common_words:
        story.append(Paragraph(f"- {word}: {count}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Daily Sentiment Trends
    story.append(Paragraph("<b>Daily Sentiment Trends</b>", styles['Heading2']))
    for log in sentiment_logs:
        date = log['Date'].strftime('%Y-%m-%d')
        start_sent = log['start_sentiment']
        end_sent = log['end_sentiment']
        most_common = log['most_common_sentiment']
        story.append(Paragraph(f"Date: {date} | Start: {start_sent} | End: {end_sent} | Most Common: {most_common}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Add Images (from BytesIO objects)
    # Note: ReportLab can take BytesIO objects directly
    story.append(Paragraph("<b>Word Cloud</b>", styles['Heading2']))
    story.append(Image(images_data['wordcloud'], width=500, height=250))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Sentiment Trend Plot</b>", styles['Heading2']))
    story.append(Image(images_data['sentiment_trend'], width=500, height=250))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Message Volumes Plot</b>", styles['Heading2']))
    story.append(Image(images_data['message_volumes'], width=500, height=250))

    # Build the PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

@app.post("/analyze")
async def analyze_chat(chat_file: UploadFile = File(...)):
    """
    Analyzes a chat file and returns a PDF report.
    
    Args:
        chat_file: The chat file to be analyzed.
    
    Returns:
        A StreamingResponse with the PDF report.
    """
    chat_name = chat_file.filename.split('_')
    chat_name = " ".join([" ".join (chat_name[3:-1]),  chat_name[-1].split(".txt")[0]])
    conversation_text = (await chat_file.read()).decode('utf-8')

    # Data Processing
    data = []
    lines = conversation_text.split('\n')
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

    df = pd.DataFrame(data, columns=["Date", "Time", "Contact", "Message"])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna()

    df["Message"] = df['Message'][df["Message"] != "<Media omitted>"]
    df.dropna(axis=0, inplace=True)
    df = df[~df["Message"].str.contains("deleted this message", case=False)]

    stop_words = set(stopwords.words('english'))
    df['Cleaned_message'] = df["Message"].apply(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)))
    df['Tokenized_words'] = df["Cleaned_message"].apply(lambda y: [word for word in word_tokenize(y) if word not in stop_words])

    lemmatizer = WordNetLemmatizer()
    df["Lemmatized"] = df["Tokenized_words"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    sentiments = SentimentIntensityAnalyzer()
    df['Positive'] = df['Message'].apply(lambda x: sentiments.polarity_scores(x)['pos'])
    df['Negative'] = df['Message'].apply(lambda x: sentiments.polarity_scores(x)['neg'])
    df['Neutral'] = df['Message'].apply(lambda x: sentiments.polarity_scores(x)['neu'])
    df['sentiment_compound'] = df['Message'].apply(lambda x: sentiments.polarity_scores(x)['compound'])
    
    def compare_values(row):
        if row['Positive'] > row['Negative'] and row['Positive'] > row['Neutral']:
            return 'positive'
        elif row['Negative'] > row['Positive'] and row['Negative'] > row['Neutral']:
            return 'negative'
        else:
            return 'neutral'
    df["sentiment"] = df.apply(compare_values, axis=1)

    grouped = df.groupby('Date')
    def most_common_sentiment(series):
        return Counter(series).most_common(1)[0][0]
    sentiment_logs = grouped.agg(
        start_sentiment=('sentiment', 'first'),
        end_sentiment=('sentiment', 'last'),
        most_common_sentiment=('sentiment', most_common_sentiment)
    ).reset_index()

    combined_words = [word for sublist in df['Lemmatized'] for word in sublist]
    fdist = nltk.FreqDist(combined_words)
    common_words = fdist.most_common(10)
    
    # Topic modeling is still done but not included in PDF to keep it concise
    df["Tokenized_mgs"] = df["Tokenized_words"].apply(lambda x: " ".join(x))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Tokenized_mgs'])
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
    df['LDA'] = relevant_topics

    # Generate plots and store them in memory as byte streams
    images_data = {}

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(fdist)
    wordcloud_io = io.BytesIO()
    wordcloud.to_image().save(wordcloud_io, 'PNG')
    wordcloud_io.seek(0)
    # images_data['wordcloud'] = "static/images/wordcloud.png"
    images_data['wordcloud'] = wordcloud_io  # Pass the BytesIO object


    df["Date"] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date
    daily_sentiment = df.groupby('Date')['sentiment_compound'].mean().reset_index()
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=daily_sentiment, x='Date', y='sentiment_compound', marker='o')
    plt.title('Sentiment Trend')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment')
    plt.grid(True)
    sentiment_io = io.BytesIO()
    plt.savefig(sentiment_io, format='png')
    sentiment_io.seek(0)
    # images_data['sentiment_trend'] = "static/images/sentiment_trend.png"
    images_data['sentiment_trend'] = sentiment_io # Pass the BytesIO object
    plt.close()

    message_counts = df.groupby('Date').size().reset_index(name='Message Count')
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=message_counts, x='Date', y='Message Count', marker='o')
    plt.title('Message Volumes Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.grid(True)
    message_io = io.BytesIO()
    plt.savefig(message_io, format='png')
    message_io.seek(0)
    # images_data['message_volumes'] = 'static/images/message_volumes.png'
    images_data['message_volumes'] = message_io
    plt.close()
    
    # Create the PDF and return it
    # pdf_buffer = create_pdf_report(common_words, sentiment_logs.to_dict('records'), images_data)

    pdf_buffer = create_pdf_report_reportlab(common_words, sentiment_logs.to_dict('records'), images_data)
    
    return StreamingResponse(
        io.BytesIO(pdf_buffer), 
        media_type="application/pdf", 
        headers={"Content-Disposition": f"attachment; filename={chat_name} chat report.pdf"}
    )