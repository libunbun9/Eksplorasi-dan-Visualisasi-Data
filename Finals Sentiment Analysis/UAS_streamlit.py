import streamlit as st

st.title("UAS Eksplorasi dan Visual Data")

st.header("Import Data Stranger Things")


st.markdown('''Data yang akan digunakan dalam UAS ini adalah data tentang sentimen tentang series Netflix berjudul 'Stranger Things' yang akan memulai season terbarunya tahun depan. Data yang diambil berjumlah 25 sentimen yang berbahasa inggiris. Sentimen yang diambil berdasarkan dari X dengan menggunakan package tweepy. Stranger Things season 1 mulai tayang di netflix pada tahun 2016. Stranger things adalah seri horror-thriller yang dibuat oleh Duffer Brothers. Stranger things season 5 adalah season final dari series stranger things. Stranger things adalah salah satu series terbaik menurut . Netflix mengatakan bahwa season sebelumnya 'Stranger Things 4' duduk di ranking no 1 sebagai series yang sering ditonton selama 4 minggu dari awal perilisan. Maka dari itu, season terbaru dari stranger things perlu diketahui sentimen dari masyarakat positif atau negatif terhadap season terbarunya. Sentimen dari masyarakat bisa memperkirakan suatu series akan sukses atau tidak. Antusias masyarakat untuk series yang baru apakah positif atau malah negatif?''')

code_tweepy = '''
import tweepy
 
 #Put your Bearer Token in the parenthesis below
 client = tweepy.Client(bearer_token='your_bearer_token')
 
 query = '#StrangerThings5 lang:en'
 tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=25)
 
 for tweet in tweets.data:
     print(tweet.text)
     if len(tweet.context_annotations) > 0:
         print(tweet.context_annotations)

import json
tweets_list=[]
for tweet in tweets.data:
    print(tweet.text)
    if len(tweet.context_annotations) > 0:
        print(tweet.context_annotations)
with open('tweets.json', 'w') as json_file:
    json.dump(tweets_list, json_file, indent=4)

tweets_list = []
for tweet in tweets.data:
    tweet_info = {
        'text': tweet.text,
        'context_annotations': tweet.context_annotations if len(tweet.context_annotations) > 0 else None
    }
    tweets_list.append(tweet_info)
with open('tweets.json', 'w') as json_file:
    json.dump(tweets_list, json_file, indent=4)
with open('tweets.json', 'r') as json_file:
    tweets_data = json.load(json_file)
import pandas as pd
# Read the JSON file
with open('tweets.json', 'r') as json_file:
    tweets_data = json.load(json_file)
    
# Convert the JSON data to a DataFrame
df = pd.DataFrame(tweets_data)    
df.to_csv("strangerthings.csv")
'''

st.code(code_tweepy)

import pandas as pd
st_ = pd.read_csv("strangerthings.csv")
st.dataframe(st_)

st.header('Preprocessing Data')
st.markdown('''
Ketika data sudah berhasil diekstrak dari X menggunakan package tweepy, data perlu dipreprocessing. Preprocessing adalah tahap membersihkan kalimat. Suatu kalimat nanti akan dipecah menjadi kata per kata. Tahap tersebut namanya tokenisasi. Preprocessing ini diperlukan untuk mengekstrak poin-poin penting dalam kalimat. Kata-kata pada tweets perlu distandarisasi menjadi satu hal yang sama. Misal semua huruf kecil, bukan kapital. Stopword removal adalah tahapan menghapus kata-kata yang tidak relevan dalam suatu kalimat dalam daftar stopword.

Tahapan preprocessing membutuhkan package nltk untuk melakukan preprocessing data.
''')

code_preprocess = '''
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Preprocessing function
def clean_text(text):
    # Remove mentions, hashtags, URLs, and special characters
    text = re.sub(r'@\w+|#\w+|http\S+|www\S+|[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_text = ' '.join(word for word in words if word not in stop_words)
    return cleaned_text

# Apply preprocessing
st['cleaned_text'] = df['text'].apply(clean_text)
'''

st.code(code_preprocess)

st.markdown('''Setelah melakukan preprocessing data, kita bisa melihat kata-kata apa yang sering muncul pada sentimen di masyarakat. untuk memvisualisasikannya bisa menggunakan word cloud.''')


code_wordcloud = '''
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all cleaned text into a single string for the word cloud
all_text = ' '.join(st['cleaned_text'])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Tweets")
plt.show()

'''

st.code(code_wordcloud)

st.image("wordcloud.png")

st.markdown('''
Dari hasil wordcloud, lumayan banyak kata-kata yang sering muncul, seperti rt, wrap party, filming dan stranger things. untuk rt sepertinya merupakan retweet dari twitter. Dari tweets ini, bisa dikatakan bahwa fans stranger things menunggu season terbarunya keluar. bisa dilihat dari kata filming, spoiler, photos, spotted dan graduation scene. Pada fans terlihat menunggu season terkahir dari stranger things. Pada wordcloud juga terdapat nama-nama actor yang memainkan stranger things season 5, seperti finn wolfhard, noah schnapp dan sadie sink. Ditemukan juga kata shipper untuk para aktor, seperti sink caleb.

Setelah data sudah dipreprocess dan lakukan visualisasi wordcloud, mari lakukan analisis sentimen. Sebelum itu, perlu adanya model analisis sentimen.
''')

code_123 = '''
df = pd.read_csv(r'C:/Users/yippi/Documents/kuliah ali/7_visual data/movie_data.csv')

import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

porter = PorterStemmer()
def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

def preprocessor(text):
    text = re.sub(r'<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

df['clean_review'] = df['review'].apply(preprocessor)
'''
st.code(code_123)

st.markdown(''' Dataset akan dibagi menjadi 80% untuk pelatihan dan 20% untuk pengujian. Data akan dibagi menggunakan package scikit-learn. Kode untuk proses praproses ada di bawah ini.''')

code_456 = '''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['clean_review'], df['sentiment'], test_size=0.2, random_state=42)'''

st.code(code_456)

st.markdown('''Model analisis sentimen yang akan digunakan adalah metode Term Frequency-Inverse Document Frequency (TF-IDF). Data teks diubah menjadi TF-IDF. TF-IDF adalah metode yang menghitung seberapa relevan suatu kata dalam rangkaian atau korpus terhadap sebuah teks. Relevansi meningkat sebanding dengan jumlah kemunculan kata dalam teks, tetapi dikompensasi oleh frekuensi kata dalam data.''')

code_789 = '''
from sklearn.feature_extraction.text import TfidfVectorizer

# Define TF-IDF vectorizer
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        tokenizer=tokenizer_porter,
                        use_idf=True,
                        norm='l2',
                        smooth_idf=True)

# Transform text data into TF-IDF vectors
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)'''

st.code(code_789)
st.markdown('''
Setelah data sudah diTF-IDF, data akan diprediksi menggunakan regresi logistik.''')

code_10 = '''
from sklearn.linear_model import LogisticRegression

# Train Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Calculate predictions on the test data
y_pred = lr.predict(X_test_tfidf)

# Evaluate using metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
'''
st.code(code_10)
st.image("evaluation.png")

st.markdown('''
Akurasi model adalah 89,56%. Model true positive adalah 88,21%. Semua prediksi true positive dalam set uji adalah 91,16%. Skor f1 adalah 0,8966.
''')

code_11 = '''
# preprocessing data
porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

def preprocessor(text):
    text = re.sub(r'<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

# Clean review text
st['clean_review'] = st['text'].apply(preprocessor)

all_data_tfidf = tfidf.transform(st['clean_review'])
all_data_predictions = lr.predict(all_data_tfidf)

# Add predictions to DataFrame
st['predicted_sentiment'] = all_data_predictions
st[['clean_review','predicted_sentiment']]
'''

st.code(code_11)

predict = pd.read_csv("st_predict.csv")
st.dataframe(predict[['clean_review','predicted_sentiment']])

sentiment_counts = predict['predicted_sentiment'].value_counts()

# Extract counts for positive and negative sentiments
negative_count = sentiment_counts.get(0, 0)
positive_count = sentiment_counts.get(1, 0)

import plotly.express as px
# Define labels and values for the pie chart
labels = [f'Negative (0): {negative_count}', f'Positive (1): {positive_count}']
sizes = [negative_count, positive_count]
fig = px.pie(values=sizes, names=labels)
fig.update_layout(
    title = "Sentiment Distribution of Stranger Things 5"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('''

Sentimen Positif pada series stranger things 5 adalah 14 sentimen (56%), dan sentimen negatif sebanyak 11 sentimen (44%). Sentimen stranger things 5 kebanyakan positif, dari hal ini dapat disimpulkan adalah sentimen untuk series stranger things sebagian besar positif.
''')

