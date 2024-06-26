import os  
import numpy as np 
import cv2 
from keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model
 
is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
        if not is_init:
            is_init = True 
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]]*size).reshape(-1,1)
        else:
            loaded_array = np.load(i)
            if len(loaded_array.shape) == 1:  # If loaded array has only one feature
                loaded_array = loaded_array.reshape((-1, 1))  # Reshape to have one feature per row
            X = np.concatenate((X, loaded_array), axis=0)  # Concatenate along samples (axis 0)
            y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c += 1

for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# Convert labels to one-hot encoding
y = to_categorical(y)

# Shuffle data
permutation = np.random.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]

# Determine the shape of the input data
input_shape = X.shape[1:]

# Define the input layer with the correct shape
ip = Input(shape=input_shape)

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X, y, epochs=50)

model.save("model.h5")
np.save("labels.npy", np.array(label))
