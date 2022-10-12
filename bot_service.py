import pickle
import random

import nltk
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

nltk.download('popular')


class Bot:
    ERROR_THRESHOLD = 0.25

    def __init__(self, user_text):
        self.lemmatizer = WordNetLemmatizer()
        self.model = load_model("Models/model.h5")
        self.words = pickle.load(open('Models/texts.pkl', 'rb'))
        self.classes = pickle.load(open('Models/labels.pkl', 'rb'))
        self.json_data = self.get_json_from_db()
        self.user_text = user_text

    @staticmethod
    def get_json_from_db():
        # url = "https://localhost:7236/api/BotConfiguration"
        # response = requests.request("GET", url, verify=False)
        # json_data = response.json()[0]
        from data_json import json_data_from_db
        json_data = json_data_from_db[0]
        return json_data

    def clean_up_sentence(self, sentence):
        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word - create short form for word
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def text_preprocessing(self, user_text, show_details=True):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(user_text)

        # bag of words - matrix of N words, vocabulary matrix
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)

    def predict(self):
        p = self.text_preprocessing(self.user_text, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        results = [[i, r] for i, r in enumerate(res) if r > self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)

        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        tag = return_list[0]['intent']

        intends = self.json_data["intents"]
        result = None
        for i in intends:
            if i['tag'] == tag:
                result = random.choice(i['responses'])

        return result
