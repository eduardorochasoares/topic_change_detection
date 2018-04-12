from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

import glob
import numpy as np
model = load_model('models_dnn/best_model.h5')

dir = "/home/eduardo/test_aula/"
#files = glob.glob(dir+"*.txt")
docs = ["" for x in range(216)]
i = 0 
for i in range(216):
	f = open(dir+"anotation"+str(i)+".txt")
	s = f.readlines()
	if(s):
		docs[i] =str(s)

for w in range(0,len(docs)):
	docs[w] = docs[w].replace(" \'","").replace("\'","").replace("[","").replace("]","")
docs = np.array(docs)
vocab_size = 5000
encoded_docs = [one_hot(d, vocab_size) for d in docs]
padded_docs = pad_sequences(encoded_docs, maxlen=20, padding='post')
y = model.predict( padded_docs, batch_size=None, verbose=0, steps=None)
#y_classes = np.utils.probas_to_classes(y)
i = 0
for i in range(len(y)):
	if y[i] > 0.9:
		y[i] = 1
	else:
		y[i] = 0
i = 0
for i in range(len(y)):
	if y[i] == 1:
		print(i)

