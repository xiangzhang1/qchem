from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking
from keras.preprocessing.sequence import pad_sequences
import numpy as np

nsamples = 3000
timesteps = 600
data_dim = 4

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(Masking(mask_value=0.0,
               input_shape=(timesteps, data_dim)))
model.add(LSTM(8, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(8))  # return a single vector of dimension 32
model.add(Dense(8, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_absolute_percentage_error',
              optimizer='adam',
              metrics=['accuracy'])

# Generate dummy training data
x_train = []
y_train = []
for i in range(1000):
    timesteps_ = np.random.randint(low=0, high=timesteps)
    x_train_ = np.random.random((timesteps_, data_dim)) - 0.5
    y_train_ = np.sum(np.linalg.norm(x_train_, axis=0), keepdims=True)
    x_train.append(x_train_)
    y_train.append(y_train_)

x_train = pad_sequences(x_train, dtype='float32', maxlen=timesteps)
y_train = np.array(y_train)

# Generate dummy validation data
x_val = []
y_val = []
for i in range(500):
    timesteps_ = np.random.randint(low=0, high=timesteps)
    x_val_ = np.random.random((timesteps_, data_dim)) - 0.5
    y_val_ = np.sum(np.linalg.norm(x_val_, axis=0), keepdims=True)
    x_val.append(x_val_)
    y_val.append(y_val_)

x_val = pad_sequences(x_val, dtype='float32', maxlen=timesteps)
y_val = np.array(y_val)

# Genreate dummy training data

model.fit(x_train, y_train,
          batch_size=64, epochs=40,
          validation_data=(x_val, y_val))

y_predict = model.predict(x_val)
model.evaluate(x_val, y_val)
plt.scatter(y_val, y_predict)
