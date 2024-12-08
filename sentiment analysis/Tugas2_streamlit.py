import streamlit as st
st.title("Sentiment Analysis")

st.markdown("First of all, Data from X (used to be called twitter) was scrapped using tweepy packages. The data was  about 50 sentiment of manchester city. ")

import pandas as pd
tweets = pd.read_csv(r'C:\Users\yippi\Documents\kuliah ali\7_visual data\pep.csv')
st.dataframe(tweets)
st.caption("Scrapped from twitter")



st.markdown("to make a sentiment analysis, we need to make the model first. In this analysis, model that will be used is from movie_data. To load the dataset, will be use Pandas Dataframe. Before begin to build the model, we need to clean the data, in this particular analysis the name is preprocessing text. Preprocessing text is a step to remove everything except the point of thesentence. Preprocessing text incluces several subtasks, such aas removing HTML tags, converst to lowercase, removing non-alphanumeric characters, extracting emoticons, and tokenizing words. Dataset will be splited into 80% of training and 20% for testing. The data will be splitted using scikit-learn. The code fot preprocessing process in below")

code_preprocessing = '''
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

# Clean review text
df['clean_review'] = df['review'].apply(preprocessor)

        ")
'''

st.code(code_preprocessing)

st.markdown("After the data already preprocessed, the data are ready to be trained into model. The data will be splitted into two, train data and testing. Usually, the test data was 80% of whole data, and the rest of it is test data. The model was build from 80% data. ")


code_traintest = '''
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_review'], df['sentiment'], test_size=0.2, random_state=42)
'''
st.code(code_traintest)

st.markdown("To make the model we will use Term Frequency-Inverse Document Frecuency or TF-IDF methods. The text data is transform into TF-IDF. TF-IDF is a method that calculate how relevant a word in a series or corpus is to a text. The meaning increases proportionally to the number of times in the text a word appears but is compensated by the word frequency in the data.")

code_tfidf = '''
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
X_test_tfidf = tfidf.transform(X_test)
'''
st.code(code_tfidf)

st.markdown("After TF-IDF, we will predict using logistic regression. we use logistic regression becouse our final output was a categorical. 1 for positive sentiment and 0 for negative sentiment")

code_trainlog = '''
from sklearn.linear_model import LogisticRegression

# Train Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)
'''
st.code(code_trainlog)

st.image(r"C:\Users\yippi\OneDrive\Pictures\Screenshots\Screenshot 2024-12-09 054121.png")

st.markdown("after the logistic regression was built, let's evaluate out model. ")

code_evaluation = '''
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
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
'''
st.code(code_evaluation)

st.image(r"C:\Users\yippi\OneDrive\Pictures\Screenshots\Screenshot 2024-12-08 225239.png")

st.markdown("The evaluation of the model: The accuracy of model is 89,56%. the true positive model is 88,21%. all the true positive predictions in the test set is 91,16%. The f1 score is 0,8966")


st.markdown("from the model above, we will use that model to predict our own sentiment analysis from X data about manchester city.")



code_predict = '''
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
tweets['clean_review'] = tweets['text'].apply(preprocessor)

all_data_tfidf = tfidf.transform(df['clean_review'])
all_data_predictions = lr.predict(all_data_tfidf)

# Add predictions to DataFrame
df['predicted_sentiment'] = all_data_predictions
'''
st.code(code_predict)

predict_sentiment = pd.read_csv(r"C:\Users\yippi\Documents\kuliah ali\7_visual data\manchestercity.csv")
st.dataframe(predict_sentiment)

st.markdown("Yayyy, we did it! we succesfully predict sentiment analysis. See you next ime :>")

st.header("References")
st.markdown(
"""
- https://github.com/zenklinov/regression_logistic_-_sentiment_analysis_movie_data
- https://github.com/zenklinov/Regression_Logistic_-_Sentiment_Analysis
- https://towardsdatascience.com/how-to-access-data-from-the-twitter-api-using-tweepy-python-e2d9e4d54978
"""
)

st.caption('''
Alicia 
Statistics Matana University
''')