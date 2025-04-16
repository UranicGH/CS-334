import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def gensim():
    gensim = Word2Vec()
    

def tfidf(train, test, k):
    vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase=True, stop_words='english', max_features=k)

    train = vectorizer.fit_transform(train)
    test = vectorizer.transform(test)

    return pd.DataFrame(train.toarray(), columns=vectorizer.get_feature_names_out()), pd.DataFrame(test.toarray(), columns=vectorizer.get_feature_names_out())

def main():
    data = pd.read_csv("youtube.csv")
    # data = data.drop('Sentiment', axis=1)

    # fill nulls
    data.dropna(inplace=True)

    train, test = train_test_split(data, train_size=0.8)

    yTrain, yTest = train['Sentiment'], test['Sentiment']

    xTrain, xTest = tfidf(train['Comment'], test['Comment'], k=2000)

    # train = pd.concat([w_train, train], axis=1)
    # test = pd.concat([w_test, test], axis=1)

    # train.drop(columns='Comment', inplace=True)
    # test.drop(columns='Comment', inplace=True)

    xTrain.to_csv("xTrain.csv", index=False)
    xTest.to_csv("xTest.csv", index=False)

    yTrain.to_csv("yTrain.csv", index=False)
    yTest.to_csv("yTest.csv", index=False)

if __name__ == "__main__":
    main()