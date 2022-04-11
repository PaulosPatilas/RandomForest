import pandas as pd
import numpy as np
import gensim
import re
from nltk.corpus import stopwords   
from nltk.stem.porter import PorterStemmer   
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim import utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding
from keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D, LSTM
from keras.callbacks import ReduceLROnPlateau, EarlyStopping



tokenizer=Tokenizer()
email_df = pd.read_csv('spam_or_not_spam.csv')
Port_stem=PorterStemmer()
corpus=[]
X=email_df ['text']
Y=email_df .label

for i in range(len(email_df['email'])):
    text_1=re.sub('[^a-zA-Z]'," ",email_df['email'].astype(str)[i])
    text_1=text_1.lower()
    text_1=text_1.split()
    text_1=[Port_stem.stem(word) for word in text_1 if word not in stopwords.words('english')]
    text_1=' '.join(text_1)
    corpus.append(text_1)

Y =  email_df['label']
xtrain,xval,ytrain,yval=train_test_split(corpus,Y,test_size=0.2,random_state=2)

documents=[text.split() for text in xtrain]

w2v_model = gensim.models.Word2Vec(vector_size=100, min_count=2, window=10, workers=8) #initiate model , min_count Ignores all words with total frequency lower than this.
w2v_model.build_vocab(documents) #build vocab from a dictionary of word frequencies.
words = w2v_model.wv.key_to_index.keys()
vocab_size = len(words)
w2v_model.train(documents,total_examples=len(documents),epochs=32)


tokenizer.fit_on_texts(xtrain)
vocab_size = len(tokenizer.word_index) + 1

x_train = pad_sequences(tokenizer.texts_to_sequences(xtrain), maxlen=100) # creating the numerical sequence matrix
x_test = pad_sequences(tokenizer.texts_to_sequences(xval), maxlen=100)

# Build Embedding Layer
embedding_matrix = np.zeros((vocab_size, 100))
print(embedding_matrix)
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i-1] = w2v_model.wv[word]

# Build The model
		
embedding_layer = Embedding(vocab_size, 16,weights=[embedding_matrix], input_length=100)	

model = Sequential()
model.add(embedding_layer) 
model.add(Dropout(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0), EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
history = model.fit(x_train, ytrain,batch_size=32,epochs=8,validation_split=0.1,verbose=1,callbacks=callbacks)