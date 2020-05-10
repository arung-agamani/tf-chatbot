import tensorflow as tf
import nltk
import json
import pickle
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        ## tokenize
        token = nltk.word_tokenize(pattern)
        words.extend(token)
        documents.append((token, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(token.lower()) for token in words if token not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pickle', 'wb'))
pickle.dump(classes, open('classes.pickle', 'wb'))

# making training data
train_data = []
output_empty = [0] * len(classes)

for document in documents:
    bag_of_words = []
    pattern_words = document[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        bag_of_words.append(1) if word in pattern_words else bag_of_words.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    train_data.append([bag_of_words, output_row])

random.shuffle(train_data)
train_data = np.array(train_data)
train_pattern = list(train_data[:,0])
train_intents = list(train_data[:,1])
print("Training data created")
# print(train_pattern)
# print(len(train_pattern[0]), "train_pattern",train_pattern[0])

# MACHINE LEARNING MODEL BABY
model = Sequential()
model.add(Dense(128, input_shape=(len(train_pattern[0]), ), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_intents[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_pattern), np.array(train_intents), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("Model created")