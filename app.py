import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
import re
import numpy as np
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Use session state to store processed data and plots
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'plots' not in st.session_state:
    st.session_state['plots'] = {}

# Ensure NLTK data is downloaded once using session state
if 'nltk_downloaded' not in st.session_state:
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    st.session_state['nltk_downloaded'] = True

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv('final_ds.csv')

# Load model evaluation data with caching
@st.cache_data
def load_model_evaluation_data():
    return pd.read_csv('modelevaluation.csv')

# Load the trained models and vectorizer from pickle files
@st.cache_resource
def load_models_and_vectorizer():
    with open('BaggingClassifier.pkl', 'rb') as file:
        bagging_model = pickle.load(file)
    with open('DecisionTreeClassifier.pkl', 'rb') as file:
        decision_tree_model = pickle.load(file)
    with open('vector_vocabulary.pkl', 'rb') as file:
        vocab = pickle.load(file)
        # Create a new CountVectorizer with the loaded vocabulary
        vectorizer = CountVectorizer(stop_words='english', 
                                   lowercase=True,
                                   vocabulary=vocab)
    return bagging_model, decision_tree_model, vectorizer

# Use the globally loaded DataFrame
df = load_data()
st.session_state['df'] = df

# Function to display toxic vs non-toxic distribution
def performDataDistribution(df):
    if 'data_distribution' not in st.session_state['plots']:
        plt.style.use('fivethirtyeight')
        total = df.shape[0]
        num_non_toxic = df[df.Hate == 0].shape[0]
        slices = [num_non_toxic / total, (total - num_non_toxic) / total]
        labeling = ['Non-Hate Speech', 'Hate Speech']
        explode = [0.05, 0.05]
        fig, ax = plt.subplots()
        ax.pie(slices, explode=explode, shadow=True, autopct='%1.2f%%', labels=labeling, wedgeprops={'edgecolor': 'black'})
        ax.set_title('Hate vs. Non-Hate Speech Proportion')
        plt.tight_layout()
        st.session_state['plots']['data_distribution'] = fig
    st.pyplot(st.session_state['plots']['data_distribution'])

# Function to clean and preprocess data with caching
@st.cache_data
def perform_data_set_cleaning(df):
    if st.session_state['processed_data'] is not None:
        return st.session_state['processed_data']
        
    def remove_pattern(input_txt, pattern):
        if isinstance(input_txt, str):
            r = re.findall(pattern, input_txt)
            for i in r:
                input_txt = re.sub(i, '', input_txt)
            return input_txt
        else:
            return ""

    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Month'] = df['Date'].dt.month
    df['lenText'] = df['text'].str.len()
    combined_df = pd.concat([df, df], ignore_index=True)
    combined_df['CleanedText'] = np.vectorize(remove_pattern)(combined_df['text'], "@[\\w]*")
    combined_df['CleanedText'] = combined_df['CleanedText'].str.replace("[^a-zA-Z#]", " ", regex=True)
    combined_df['lenCleanedText'] = combined_df['CleanedText'].str.len()
    combined_df['CleanedText'] = combined_df['CleanedText'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
    tokenized_tweet = combined_df['CleanedText'].apply(lambda x: x.split())
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    combined_df['CleanedText'] = tokenized_tweet.apply(' '.join)
    st.session_state['processed_data'] = combined_df
    return combined_df

# Function to plot word clouds
def plot_word_clouds(combined_df):
    if 'word_clouds' not in st.session_state['plots']:
        combined_df['text'] = combined_df['text'].astype(str)
        hate_text = ' '.join(combined_df[combined_df['Hate'] == 1]['text'])
        non_hate_text = ' '.join(combined_df[combined_df['Hate'] == 0]['text'])
        wc_hate = WordCloud(width=800, height=400, background_color='white').generate(hate_text)
        wc_non_hate = WordCloud(width=800, height=400, background_color='white').generate(non_hate_text)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(wc_hate, interpolation='bilinear')
        ax[0].axis('off')
        ax[0].set_title('Hate Speech')
        ax[1].imshow(wc_non_hate, interpolation='bilinear')
        ax[1].axis('off')
        ax[1].set_title('Non-Hate Speech')
        st.session_state['plots']['word_clouds'] = fig
    st.pyplot(st.session_state['plots']['word_clouds'])

# Function to plot sentiment distribution
def plot_sentiment_distribution(combined_df):
    if 'sentiment_distribution' not in st.session_state['plots']:
        sia = SentimentIntensityAnalyzer()
        combined_df['Sentiment'] = combined_df['CleanedText'].apply(lambda x: sia.polarity_scores(x)['compound'])
        fig, ax = plt.subplots()
        sns.histplot(combined_df, x='Sentiment', hue='Hate', multiple='stack', bins=30, ax=ax)
        ax.set_title('Sentiment Distribution Hate vs. Non-Hate Speech')
        st.session_state['plots']['sentiment_distribution'] = fig
    st.pyplot(st.session_state['plots']['sentiment_distribution'])

# Function to plot hate speech distribution by month
def plot_hate_speech_by_month(combined_df):
    if 'hate_speech_by_month' not in st.session_state['plots']:
        combined_df['Month'] = combined_df['Date'].dt.month_name()
        fig, ax = plt.subplots()
        sns.countplot(x='Month', hue='Hate', data=combined_df, order=combined_df['Month'].value_counts().index.tolist(), ax=ax)
        ax.set_title('Hate Speech Distribution by Month')
        plt.xticks(rotation=45)
        st.session_state['plots']['hate_speech_by_month'] = fig
    st.pyplot(st.session_state['plots']['hate_speech_by_month'])

# Function to plot hate speech distribution by time category
def plot_hate_speech_by_time_category(combined_df):
    if 'hate_speech_by_time_category' not in st.session_state['plots']:
        fig, ax = plt.subplots()
        sns.countplot(x='TimeCategory', hue='Hate', data=combined_df, ax=ax)
        ax.set_title('Hate Speech Distribution by Time Category')
        plt.xticks(rotation=45)
        st.session_state['plots']['hate_speech_by_time_category'] = fig
    st.pyplot(st.session_state['plots']['hate_speech_by_time_category'])

# Function to plot hate speech by hour of day
def plot_hate_speech_by_hour(combined_df):
    if 'hate_speech_by_hour' not in st.session_state['plots']:
        combined_df['Hour'] = pd.to_datetime(combined_df['Time'], format='%H:%M:%S').dt.hour
        fig, ax = plt.subplots()
        sns.histplot(data=combined_df, x='Hour', hue='Hate', multiple='stack', binwidth=1, ax=ax)
        ax.set_title('Hate Speech by Hour of Day')
        ax.set_xlabel('Hour')
        st.session_state['plots']['hate_speech_by_hour'] = fig
    st.pyplot(st.session_state['plots']['hate_speech_by_hour'])

# Function to plot top 10 dates with most hate speech
def plot_top_10_hate_dates(combined_df):
    if 'top_10_hate_dates' not in st.session_state['plots']:
        hate_speech_combined_df = combined_df[combined_df['Hate'] == 1]
        hate_by_date = hate_speech_combined_df.groupby('Date').size().reset_index(name='HateCount')
        top_10_hate_dates = hate_by_date.sort_values(by='HateCount', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Date', y='HateCount', data=top_10_hate_dates, ax=ax)
        ax.set_title('Top 10 Dates with Most Hate Speech')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Hate Speech Instances')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.session_state['plots']['top_10_hate_dates'] = fig
    st.pyplot(st.session_state['plots']['top_10_hate_dates'])

# Function to plot top 20 words in hate and non-hate speech
def plot_top_20_words(combined_df):
    if 'top_20_words' not in st.session_state['plots']:
        hate_words = ' '.join(combined_df[combined_df['Hate'] == 1]['CleanedText']).split()
        non_hate_words = ' '.join(combined_df[combined_df['Hate'] == 0]['CleanedText']).split()
        hate_freq = pd.DataFrame(Counter(hate_words).most_common(20), columns=['Word', 'Frequency'])
        non_hate_freq = pd.DataFrame(Counter(non_hate_words).most_common(20), columns=['Word', 'Frequency'])
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.barplot(x='Frequency', y='Word', data=hate_freq, ax=ax[0])
        ax[0].set_title('Top 20 Words in Hate Speech')
        sns.barplot(x='Frequency', y='Word', data=non_hate_freq, ax=ax[1])
        ax[1].set_title('Top 20 Words in Non-Hate Speech')
        plt.tight_layout()
        st.session_state['plots']['top_20_words'] = fig
    st.pyplot(st.session_state['plots']['top_20_words'])

# Function to plot text length distribution
def plot_text_length_distribution(combined_df):
    if 'text_length_distribution' not in st.session_state['plots']:
        fig, ax = plt.subplots()
        sns.histplot(combined_df, x='lenText', hue='Hate', multiple='stack', bins=30, ax=ax)
        ax.set_title('Text Length Distribution in Hate vs. Non-Hate Speech')
        ax.set_xlabel('Length of Text')
        st.session_state['plots']['text_length_distribution'] = fig
    st.pyplot(st.session_state['plots']['text_length_distribution'])

# Function to plot n-grams
def plot_ngrams(combined_df, n=2):
    key = f'{n}_grams'
    if key not in st.session_state['plots']:
        vectorizer = CountVectorizer(ngram_range=(n, n))
        ngrams = vectorizer.fit_transform(combined_df[combined_df['Hate'] == 1]['CleanedText'])
        ngrams_freq = pd.DataFrame(ngrams.sum(axis=0), columns=vectorizer.get_feature_names_out()).T
        ngrams_freq.columns = ['Frequency']
        ngrams_freq = ngrams_freq.sort_values(by='Frequency', ascending=False).head(20).reset_index()
        ngrams_freq.columns = ['Ngram', 'Frequency']
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='Frequency', y='Ngram', data=ngrams_freq, ax=ax)
        ax.set_title(f'Top 20 {n}-grams in Hate Speech')
        plt.tight_layout()
        st.session_state['plots'][key] = fig
    st.pyplot(st.session_state['plots'][key])

# Function to display machine learning results
def display_ml_results():
    st.header("Machine Learning Results")
    st.markdown("""
    > This section will display the results of the machine learning models used for hate speech detection.
    """)
    # Load and display the model evaluation data
    model_evaluation_df = load_model_evaluation_data()
    st.dataframe(model_evaluation_df)
    
    # Display images with explanations
    st.image("ROC Curves For All Models.png", caption="ROC Curves for All Models")
    st.markdown("""
    > The ROC curves show the performance of the classification models. A higher area under the curve (AUC) indicates better performance.
    """)
    
    st.image("Trauining and Prediction Time.png", caption="Training and Prediction Time")
    st.markdown("""
    > This chart displays the training and prediction times for each model. Faster times are generally preferred, but must be balanced with accuracy.
    """)
    
    st.image("ClassiificationSummary.png", caption="Classification Summary")
    st.markdown("""
    > The classification summary provides key metrics such as accuracy, precision, recall, and F1 score for each model.
    """)
    
    st.image("Confusion Matrix.png", caption="Confusion Matrix")
    st.markdown("""
    > The confusion matrix shows the number of true positive, true negative, false positive, and false negative predictions for each model.
    """)

    st.subheader("BERT Model")

    st.image("ConfusionMatrix BERT.png", caption="BERT Confusion Matrix")
    st.markdown(""">The following is Confusion Matrix for our BERT model. It shows the number of true positive, true negative, false positive, and false negative predictions for each model.""")


# Function to preprocess text for prediction
def preprocess_text(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", "", text)
    text is text.lower()
    text = ' '.join([word for word in text.split() if len(word) > 3])
    lemmatizer = nltk.stem.WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Testing new data
def test_new_data(text):
    count_vector = CountVectorizer(stop_words='english', lowercase=True, vocabulary=pickle.load(open("vector_vocabulary.pkl", "rb")))
    data = count_vector.transform([text])

    results = {}
    for model_name in ['RandomForestClassifier']:
        model = pickle.load(open(model_name + ".pkl", 'rb'))
        prediction = model.predict(data)
        results[model_name] = "Hate Speech" if prediction == 1 else "Non-Hate Speech"
    
    return results

# Updated hate_speech_detector function with error handling
def hate_speech_detector():
    st.header("Hate Speech Detector")
    st.markdown("""
    > Enter text to detect if it contains hate speech.
    """)
    user_input = st.text_area("Enter text here:", key="hate_speech_detector")
    
    if st.button("Detect", key="hate_speech_detector_button"):
        if user_input:
            try:
                # Get predictions
                results = test_new_data(user_input)
                # Display results
                st.subheader("Machine Learning Models")
                for model_name, prediction in results.items():
                    st.write(f"**{model_name} Prediction:** {prediction}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text to detect.")
    


def BERT_test(bert_input):
    # Testing on a single example
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) 
    model.load_state_dict(torch.load('saved_complete_bert_model.pth', map_location=torch.device('cpu'))) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_text = bert_input 
    encoded_test = tokenizer(
        test_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids_test = encoded_test['input_ids'].to(device)
    attention_mask_test = encoded_test['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids_test, attention_mask=attention_mask_test)
        prediction = torch.argmax(output.logits, dim=1).item()

    return "Hate" if prediction == 1 else "Not Hate"

def BERT_hate_speech_detector():
    st.header("BERT Hate Speech Detector")
    st.markdown("""
    > Enter text to detect if it contains hate speech.
    """)
    user_input = st.text_area("Enter text here:", key="bert_hate_speech_detector")
    
    if st.button("Detect", key="bert_hate_speech_detector_button"):
        if user_input:
            try:
                # Get predictions
                prediction = BERT_test(user_input)
                
                # Display results
                st.subheader("BERT Model")
                st.write(f"**Prediction:** {prediction}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text to detect.")

# Main execution
st.title("Hate Speech Detection Dashboard")

if df is not None:
    # Get or process data
    combined_df = perform_data_set_cleaning(df)
    
    # Create tabs for better organization and performance
    tabs = st.tabs([
        "Data Distribution",
        "Word Clouds",
        "Sentiment Analysis",
        "Time Analysis",
        "Word Analysis",
        "Machine Learning Results",
        "Hate Speech Detector"
    ])
    
    with tabs[0]:
        st.header("Data Distribution")
        performDataDistribution(df)
        st.markdown("""
        >This pie chart shows the proportion of hate speech versus non-hate speech in the dataset.We observe that 64.40 percent of the data is hate speech, while 35.60 percent is Non-hate speech.
        """)
    
    with tabs[1]:
        st.header("Word Clouds")
        plot_word_clouds(combined_df)
        st.markdown("""
        >Word clouds provide a visual representation of the most frequent words.Word seen above are most used word in the dataset.
        """)
    
    with tabs[2]:
        st.header("Sentiment Distribution")
        plot_sentiment_distribution(combined_df)
        st.markdown("""
        >This histogram shows the distribution of sentiment scores. We observe that the sentiment scores are more negative for hate speech compared to non-hate speech.
        """)
    
    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Monthly Distribution")
            plot_hate_speech_by_month(combined_df)
            st.write(">Hate speech is more prevalent in the months of October and July.")
        with col2:
            st.subheader("Time Category Distribution")
            plot_hate_speech_by_time_category(combined_df)
            st.write(">Hate speech is more prevalent in the night time.")
        plot_hate_speech_by_hour(combined_df)
        st.write(">Hate speech is more prevalent during the 24th Hour.")
        plot_top_10_hate_dates(combined_df)
        st.write(">The top 10 dates with the most hate speech are shown above.We observe there is enormous increase in hate speech on 2023-10-25.This could be due to targetted Hate Speech Attack.")
        st.image("Temporal Analysis.png", caption="Classification Summary")
        
    
    with tabs[4]:
        st.header("Word Analysis")
        plot_top_20_words(combined_df)
        st.write(">The top 20 words in hate speech and non-hate speech are shown above. We observe that the most frequent words in hate speech are more negative and offensive.")
        plot_text_length_distribution(combined_df)
        st.write(">The text length distribution in hate speech and non-hate speech is shown above. We observe that hate speech tends to have shorter text lengths.")
        plot_ngrams(combined_df, n=2)
        st.write(">The top 20 bigrams in hate speech are shown above. We observe that the bigrams are more negative and offensive.")
        plot_ngrams(combined_df, n=3)
        st.write(">The top 20 trigrams in hate speech are shown above. We observe that the trigrams are more negative and offensive.")
    with tabs[5]:
        display_ml_results()
    
    with tabs[6]:
        hate_speech_detector()
        BERT_hate_speech_detector()
