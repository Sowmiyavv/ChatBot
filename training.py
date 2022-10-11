import random
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


class BotTrainer:

    def __init__(self):
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!']
        self.lemmatizer = WordNetLemmatizer()
        self.json_data = self.get_json_from_db()

    @staticmethod
    def get_json_from_db():
        # url = "https://localhost:7236/api/BotConfiguration"
        # response = requests.request("GET", url, verify=False)
        # json_data = response.json()[0]
        from data_json import json_data_from_db
        json_data = json_data_from_db[0]
        return json_data

    def train(self):
        for intent in self.json_data['intents']:
            for pattern in intent['patterns']:

                # tokenize each word
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                # add documents in the corpus
                self.documents.append((w, intent['tag']))

                # add to our classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # lemmaztize and lower each word and remove duplicates
        words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        words = sorted(list(set(words)))

        # sort classes
        classes = sorted(list(set(self.classes)))

        # documents = combination between patterns and intents
        print(len(self.documents), "documents")

        # classes = intents
        print(len(classes), "classes", classes)

        # words = all words, vocabulary
        print(len(words), "unique lemmatized words", words)

        pickle.dump(words, open('Models/texts.pkl', 'wb'))
        pickle.dump(classes, open('Models/labels.pkl', 'wb'))

        # create our training data
        training = []
        # create an empty array for our output
        output_empty = [0] * len(classes)
        # training set, bag of words for each sentence
        for doc in self.documents:
            # initialize our bag of words
            bag = []

            # list of tokenized words for the pattern
            pattern_words = doc[0]

            # lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]

            # create our bag of words array with 1, if word match found in current pattern
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            # output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1

            training.append([bag, output_row])

        # shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training)

        # create train and test lists. X - patterns, Y - intents
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        print("Training data created")

        # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains
        # number of neurons equal to number of intents to predict output intent with softmax
        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this
        # model
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # fitting and saving the model
        hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
        model.save('Models/model.h5', hist)

        print("model created")
