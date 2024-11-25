import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences





path ='ready_data.csv'
df0=pd.read_csv(path, encoding='utf-8')

df= df0.groupby('Score').apply(lambda x: x.sample(20000)).reset_index(drop=True)

df['Text+Summary']=df['cleaned_Summary']+' '+df['cleaned_text']
df['Text+Summary'] = df['Text+Summary'].astype(str)

#print(df['cleaned_Summary'])
reviews=df['cleaned_text'].values
sentiments=df['Score'].values

# Apply tokenization
tokenizer=Tokenizer(num_words=20000)
tokenizer.fit_on_texts(reviews)
sequences=tokenizer.texts_to_sequences(reviews)

# Padding sequences on reviews to have same length
max_len = 150  # Adjusted max length as reviews are longer
# X as input
X = pad_sequences(sequences, maxlen=max_len)
# Convert sentiments to numpy array
y = np.array(sentiments)






