import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import os

## add json file
with open("intents.json") as file:
    data = json.load(file)

## if we have already data then no need to run again
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
## otherwise it need        
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)  #Every senetence broken in token
            words.extend(wrds)
            docs_x.append(wrds) ## add each token
            docs_y.append(intent["tag"]) # Add tag 

        if intent["tag"] not in labels:
            labels.append(intent["tag"]) # add in token  also

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]  # no need to consider of ?
    words = sorted(list(set(words))) # no need to consider duplicate words

    labels = sorted(labels) # no need to consider duplicates tags

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))] #empty list

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:  #if words exist in wrds then count frequency of words
                bag.append(1)
            else:
                bag.append(0)#otherwise give 0

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:       #otherwise save .pickle file
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])  #connected each word to another all words
net = tflearn.fully_connected(net, 8) # 8 nodes
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #6 nodes and use softmaxfunction which gives  prob of each word
net = tflearn.regression(net)

model = tflearn.DNN(net) #deep neutral network 

if os.path.exists("model.tflearn.meta"):   #save model training data
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):   #check prob of words and give max prob index of word
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)  # return max occured word


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        #print(tag)
        #print(result)  #gives probabilities
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

        if inp.lower() == "quit" or inp.lower() == "goodbye" or inp.lower() == "bye":
            break

chat()