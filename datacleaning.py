import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords')
stop = set(stopwords.words("english"))


def data_cleaning():
    data = pd.read_csv('200kdata.csv')
    # preprocess sentences
    print("preprocessing sentences...")
    preprocessed_df = pd.DataFrame(columns=['Data'])
    for i in range(len(data)):
        s = str(data.iloc[i][0])
        text_input = re.sub('[^a-zA-Z1-9]+', ' ', s)
        output = re.sub(r'\d+', '', text_input)
        preprocessed_df.loc[len(preprocessed_df)] = output.lower().strip()

    # remove stopwords (low-value words)
    print("removing stopwords...")
    stopwords_df = pd.DataFrame(columns=['Data'])
    for i in range(len(preprocessed_df)):
        s = str(preprocessed_df.iloc[i][0])
        filtered_words = [word.lower() for word in s.split() if word.lower() not in stop]
        stopwords_df.loc[len(stopwords_df)] = " ".join(filtered_words)

    # remove strings with <5 words
    print("removing strings with <5 words...")
    new_df = pd.DataFrame(columns=['Data'])
    for i in range(len(stopwords_df)):
        s = str(stopwords_df.iloc[i][0])
        parsed = s.split()
        if len(parsed) > 5:
            new_df.loc[len(new_df)] = s

    print("replacing unicode...")
    final_df = pd.DataFrame(columns=['Data'])
    for i in range(len(new_df)):
        # replace &#39; with '
        s = str(new_df.iloc[i][0])
        s = s.replace('&#39;', '\'')
        # replace &amp with &
        s = s.replace('&amp', '&')
        # discard urls and unicode strings
        if not 'https://' in s and not '<U' in s:
            final_df.loc[len(final_df)] = s

    print(final_df)
    final_df.to_csv('200kdatacleaned.csv', index=False)
