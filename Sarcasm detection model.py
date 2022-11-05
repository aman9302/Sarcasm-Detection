##### Aman Nair R
##### Sarcasm detection model

### Importing required libraries
# -
import tensorflow as tf
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from sklearn.model_selection import train_test_split

data = requests.get('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json')  # Downloading the data  
print(data.text[0:450])

### Storing the data in two separate arrays (X variable and y variable)
# -
sentences = []
labels = []
for item in data.json():
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
print(pd.DataFrame({'sentence' : sentences[0:10], 'label':labels[0:10]}))

X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.33, random_state=42)

### Setting tokenizer properties
# - 
vocab_size = 10000
oov_tok = "<oov>"

### Tokenizing the training data
# - 
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index  

### Defining variables for the padding properties
# -
max_length = 100
trunc_type='post'
padding_type='post'

### Creating padded sequences from train and test data
# -
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)

X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

### Setting the model parameters
# - 
embedding_dim = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

### Converting train and test lists to numpy arrays
# - 
X_train_padded = np.array(X_train_padded)
y_train = np.array(y_train)
X_test_padded = np.array(X_test_padded)
y_test = np.array(y_test)

### Training the model
# - 
num_epochs = 30
history = model.fit(X_train_padded, y_train, epochs=num_epochs, validation_data=(X_test_padded, y_test), verbose=2)

### Using the model to predict on user input text
# - 
user_input1 = input ("Enter a sentence - ")
user_input2 = input ("Enter a sentence - ")
sentence = [user_input1, user_input2] 
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))
