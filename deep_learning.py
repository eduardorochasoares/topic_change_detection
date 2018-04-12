import glob
import random
from keras import optimizers
import numpy as np
from numpy import array
from keras.layers import Input, Dense, concatenate, Activation
from sklearn.model_selection import StratifiedShuffleSplit

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, MaxPooling1D, SpatialDropout1D, GlobalMaxPooling1D, GRU
from tensorflow.python.client import device_lib
from keras.models import load_model
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
import random
print(device_lib.list_local_devices())


neg = "/home/eduardo/neg/"

pos = "/home/eduardo/pos/"

docs_neg =[]
neg_files = glob.glob(neg+"*.txt")

for w in neg_files:
	f = open(w)
	s = f.readlines()
	if(s):
		docs_neg.append(str(s))
s_neg = len(docs_neg)
w = 0	
'''
index_neg_train = []
while (len(index_neg_train) < len(docs) ):
	r = random.randint(0, 8098)2018-04-03 17:19:08.399312: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA



	if r not in index_neg_train:
		try:
			index_neg_train.append(r)
		
			f = open(neg+str(r)+".txt")
			docs.append(str(f.readlines()))
		except IOError:
			print("aaa")
'''
#pos_files = glob.glob(pos+"*.txt")
index_pos_train = []

docs_pos = []
while (len(index_pos_train) < s_neg):
	r = random.randint(1, 8098)
	if r not in index_pos_train:
		#print(len(index_pos_train))
		try:
			index_pos_train.append(r)
			f = open(pos+str(r)+".txt")
			t = f.readlines()
			#f.readlines()
			#print(d)

			if t:
				docs_pos.append(str(t))
		except IOError:
			print("aaa")
docs = docs_neg + docs_pos

w = 0	
for w in range(0,len(docs)):
	docs[w] = docs[w].replace(" \'","").replace("\'","").replace("[","").replace("]","")
labels = [0 for x in range(0, s_neg)] +  [1 for h in range(s_neg, len(docs))]
print(labels)

#for i in range(0, s_neg):
	#labels.append(0)
#for i in range(s_neg, len(docs)):
	#labels.append(1)
labels = np.array(labels)
docs = np.array(docs)
vocab_size = 5000
encoded_docs = [one_hot(d, vocab_size) for d in docs]

max_length = 20
lstm_out = 128
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
#seed = 7
#seed = random.seed(a=None)
#print(seed)

cvscores = []
#kfold = StratifiedKFold(n_splits=10, shuffle=True)
i = 0
sd = 7348
#np.random.seed(seed=sd)
#sss= StratifiedShuffleSplit(n_splits=10, test_size=0.5)
sss= StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
units = 128

for train, test in sss.split(padded_docs, labels):
	embedding_vector_length = 300
	a = Input(shape=(max_length,))
	input_layer = Embedding(vocab_size, embedding_vector_length, input_length=max_length)(a)
	bigram_branch = Convolution1D(filters=20, kernel_size=2, padding='valid', activation='relu', strides=1)(input_layer)
	bigram_branch = GlobalMaxPooling1D()(bigram_branch)
	#bigram_branch = Dropout(0.2)(bigram_branch)
	#trigram_branch = Convolution1D(filters=20, kernel_size=3, padding='same', activation='relu', strides=1)(input_layer)
	#trigram_branch = GlobalMaxPooling1D()(trigram_branch)
	#trigram_branch = Dropout(0.3)(trigram_branch)
	
	#merged = concatenate([bigram_branch,trigram_branch], axis=1)
	
	merged = Dense(1024, activation='relu')(bigram_branch)
	merged = Dropout(0.8)(merged)
	merged = Dense(256, activation='relu')(merged)
	merged = Dropout(0.8)(merged)
	merged = Dense(128,activation='relu')(merged)
	merged = Dropout(0.8)(merged)
	merged = Dense(64, activation='relu')(merged)
	merged = Dropout(0.8)(merged)
	merged = Dense(1)(merged)
	
	output = Activation('tanh')(merged)

	model = Model(inputs=[a], outputs=[output])
	
	#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)


	model.compile(loss='binary_crossentropy',
		          optimizer='adam',
		          metrics=['accuracy'])

	#print(model.summary())
	'''model = Sequential()
	model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_length))
	model.add(Convolution1D(64, 3,padding='valid', activation='relu', strides=1))
	model.add(MaxPooling1D(pool_size=2, strides=1))
	model.add(Convolution1D(32, 2, padding='valid',  activation='relu', strides=1))
	model.add(MaxPooling1D(pool_size=2, strides=1))

	#model.add(MaxPooling1D(pool_size=2, strides=1))
	#model.add(Convolution1D(16, 4, padding='valid',  activation='relu',  strides=1))
	#model.add(MaxPooling1D(pool_size=2, strides=1))
	model.add(Flatten())
	#model.add(Dense(500,activation='relu'))
	
	model.add(Dense(1024,activation='relu'))
	model.add(Dropout(0.9))
	model.add(Dense(256,activation='relu'))
	model.add(Dropout(0.9))
	model.add(Dense(128,activation='relu'))
	model.add(Dropout(0.9))
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.9))
	model.add(Dense(1,activation='tanh'))
	
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])'''
	'''
	model = Sequential()
	model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_length))
	model.add(GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros'))
	model.add(Dense(1000,activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1,activation='sigmoid'))
	tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])'''
	tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
	model.fit(padded_docs[train], labels[train], epochs=20, callbacks=[tensorBoardCallback], batch_size=128)
	model.save('my_model'+str(i)+'.h5')
	scores = model.evaluate(padded_docs[test],labels[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
	i = i + 1
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
