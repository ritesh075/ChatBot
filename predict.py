#Library
import numpy as np
import json
import random
from myNeuron import *
from architecture import *
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize

#Data
with open("traindata.json") as f:
    intents=json.load(f)

#NLP functions
def tokenize(sentence):
    return word_tokenize(sentence)

def stem(word):
    return PorterStemmer().stem(word.lower())

def bag_of_words(tockenize_sentence,all_words):
    tockenize_sentence=[stem(w) for w in tockenize_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tockenize_sentence:
            bag[idx]=1.0
    return bag

#Data preperation
all_words = []
tags = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

def respond(input_sentence=""):
    bot_name = "Bot"
    print("Let's chat! (type 'quit' to exit)")
    sentence = input_sentence
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    output = model.feed_forward(x)
    predicted = np.argmax(output)
    tag = tags[predicted]
    probs = softmax(output)
    prob = probs[predicted]

    if prob > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # print(f"{bot_name}: {random.choice(intent['responses'])}")
                return f"{random.choice(intent['responses'])}"
    else:
        # print(f"I do not understand...")
        return f"I do not understand..."
        