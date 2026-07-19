import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# first-time
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Remove hashtag symbol and keep the text
    text = text.replace("#", "")
    # Remove numbers as they are not useful for sentiments
    text = re.sub(r"\d+", "", text)
    # Remove punctuation symbols are not useful for sentiments
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords and lemmatize
    words = []
    for word in text.split():
        if word not in stop_words:
            words.append(lemmatizer.lemmatize(word))

    return " ".join(words)


def main():
    df = pd.read_csv("data/raw/Tweets.csv")
    df = df[["text", "airline_sentiment"]]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["text"] = df["text"].apply(clean_text)
    df.to_csv("data/processed/cleaned_tweets.csv", index=False)
    #for cheaking
    print("✅ Cleaned dataset saved successfully!")
    print(df.head())

if __name__ == "__main__":
    main()