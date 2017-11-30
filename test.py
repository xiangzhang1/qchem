import random, string
from cStringIO import StringIO

from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking
from keras.preprocessing.sequence import pad_sequences

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

# sgd is bullshit. always gives average.
loss = 'mean_absolute_percentage_error'
optimizer = 'rmsprop'
epochs = 100



Sequential([
    Masking(mask_value=0.0, input_shape=(26, 4)),
    LSTM(24, return_sequences=True),
    LSTM(8),
    Dense(8, activation='relu'),
    Dense(1)
]),
# Sequential([
#     Masking(mask_value=0.0, input_shape=(600, 4)),
#     Dense(6, activation='relu'),
#     LSTM(8),
#     Dense(6, activation='relu'),
#     Dense(1)
# ]),

model.reset_states()
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])
model.fit(X, y0, batch_size=64, epochs=epochs, shuffle=True)
y = model.predict(X)

filename = randomword(6)

fig, ax = plt.subplots(1)
ax.scatter(y0.flatten(), y.flatten())
# fig.savefig(filename+'.png')

# sys.stdout = open(filename+'.txt','w')
# print 'loss: %s\noptimizer: %s\nepochs: %s\nmodel: \n' %(loss, optimizer, epochs)
# model.summary()
# sys.stdout.flush()
# sys.stdout.close()
# sys.stdout = sys.__stdout__
