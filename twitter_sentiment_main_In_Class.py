nimport pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout ,GlobalMaxPool1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import RMSprop
from joblib import dump
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping



# Load and preprocess the dataset
data = pd.read_csv('Twitter_Data.csv')

data = data.dropna()

data = data.reset_index(drop = True)

# Convert 'clean_text' column to strings
data['clean_text'] = data['text'].astype(str)
data['clean_text'] = data['clean_text'].str.replace(r'[^a-zA-Z\s]', '').str.lower() # Removal of non-text Data

#convert label string to categorical
data['label_id'] = data['sentiment'].factorize()[0]
cat_id = data[['sentiment', 'label_id']].drop_duplicates().sort_values('label_id')
cat_to_id = dict(cat_id.values)
id_to_cat = dict(cat_id[['label_id', 'sentiment']].values)

#show data id_to_kategori
id_to_cat

y = data['label_id'].values
X = data['clean_text'].values

print('Value of [label]:', y, "\n")
print('Value of [text]:', X)


# Check unique values in the 'sentiment' column
unique_sentiments = np.unique(y)
print("Unique Sentiments:", unique_sentiments)

# Ensure that your labels are numeric
#y = y.replace({'negative': 0, 'neutral': 1, 'positive': 2}) # Converting text to numerics

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize and pad text sequences
#max features
maxfeatures = 10000
tokenizer = Tokenizer(num_words=maxfeatures)
tokenizer.fit_on_texts(X_train)

#max sequential per word in NN
maxseqlen = max([len(i.split()) for i in X])
print(maxseqlen)


X_train_seq = tokenizer.texts_to_sequences(X_train)  # Tokenizing Train Data
X_test_seq = tokenizer.texts_to_sequences(X_test)    # Tokenizinf Test Data


X_train_pad = pad_sequences(X_train_seq, maxlen=maxseqlen) # Ensuring that length of rows remains same
X_test_pad = pad_sequences(X_test_seq, maxlen=maxseqlen)

X_train_pad = np.array(X_train_pad)
X_test_pad = np.array(X_test_pad)

y_train = to_categorical(y_train, num_classes = 3)
y_test = to_categorical(y_test, num_classes = 3)

y_train = np.array(y_train)
y_test = np.array(y_test)

# Build a simple LSTM model with 3 output units
#architecture model
model = Sequential()
model.add(Embedding(input_dim = maxfeatures, output_dim = 128, input_length = maxseqlen))
model.add(LSTM(128, return_sequences = True))
model.add(GlobalMaxPool1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))

opt = RMSprop(learning_rate=0.001, rho=0.7, momentum=0.5)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#declare checkpoint variable and early stopping to get best model
early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 3)


# Train the model with a reduced batch size
from datetime import datetime

#training model
start_time = datetime.now()
history = model.fit(X_train_pad, y_train,
                    batch_size = 1024, epochs = 10, shuffle = True,
                    validation_data=(X_test_pad,y_test) , verbose = 1,
                    callbacks=early_stop)
end_time = datetime.now()
print("Time out: {}".format(end_time - start_time))

# generate confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report

target = ['neu', 'neg', 'pos']
Y_pred = model.predict(X_test_pad)
Y_pred = np.argmax(Y_pred, axis=1)
Y_act = np.argmax(y_test, axis=1)
print(confusion_matrix(Y_act, Y_pred))
print(classification_report(Y_act, Y_pred, target_names = target))

# Save the trained model
model.save('Sentiment_DL_Model.h5')
dump(tokenizer, 'Sentiment_tokenizer.joblib')
