# MlPbSOptXRNN
# ==============================================================================

from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking
from keras.preprocessing.sequence import pad_sequences

class MlPbSOptXRNN(object):

    def __init__(self):
        # data
        self._X = []
        self._y0 = []
        # pipeline
        self.X_pipeline = StandardScaler()
        self.y_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('15', FunctionTransformer(func=lambda x: x * 15, inverse_func=lambda x: x / 15))
        ])
        # ann
        self.timesteps = 600
        self.data_dim = 4
        self.model = Sequential(
            Masking(mask_value=0.0, input_shape=(self.timesteps, self.data_dim)),
            LSTM(8, return_sequences=True),
            LSTM(8),
            Dense(8, activation='relu'),
            Dense(24, activation='relu'),
            Dense(3)
        )
        self.model.compile(loss='mean_absolute_percentage_error',
                      optimizer='adam',
                      metrics=['accuracy'])


    def parse_X1(self, cell):
        '''
        ccoor is coordinates. natom0 is # of Pb atoms (for determining sgn).
        returns a list.
        '''
        ccoor = cell.ccoor
        natom0 = cell.stoichiometry.values()[0]
        X1 = []
        for i, c in enumerate(ccoor):
            dcoor = ccoor - c
            sgn = np.sign((i - natom0 + 0.5) * (np.arange(len(ccoor)) - natom0 + 0.5))
            dcoorp = np.concatenate((dcoor, np.c_[sgn]), axis=1)
            X1.append(np.delete(dcoorp, i, axis=0))
        return X1

    def parse_y0(self, vasp):
        return vasp.optimized_cell.ccoor - vasp.node().cell.ccoor

    def parse_train(self, vasp):
        '''More of a handle.'''
        self._X1 += list(self.parse_X1(vasp.node().cell))
        self._y0 += list(self.parse_y0(vasp))

    def train(self):
        # pipeline
        _X = self.X_pipeline.fit_transform(pad_sequences(self._X1, dtype='float32', maxlen=self.timesteps))
        _y0 = self.y_pipeline.fit_transform(self._y0)
        model = self.model
        # fit
        model.fit(_X, _y0, batch_size=64, epochs=500)
        _y = model.predict(_X)
        IPython.embed()
