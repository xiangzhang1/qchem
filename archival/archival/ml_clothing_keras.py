# Ml_clothing
class Ml_clothing(object):

    def __init__(self):
        '''
        5 in: temperature (285+-20 C), humidity (68+-15%), pressure (1000 +- 100), windspeed (/10),
        4 out: 秋衣/T恤, 衬衣, 保暖内衣, 毛衣 (0,1)
        '''
        # initialize model
        self.X_scaler = np.float_([[285, 68, 1000, 0], [20, 15, 100, 10]])
        self.model = Sequential([
            Dense(4, activation='relu', input_dim=4),
            Dropout(0.05),
            Dense(4, activation='relu'),
            Dropout(0.05),
            Dense(4, activation='relu')
        ])
        self.model.compile(optimizer='rmsprop',
                      loss='mse')
        # train
        self.X = np.float_([])    # Y for high dimensional output, y for 1d.
        self.Y = np.float_([])    # original data is stored

    def commit(self, Y_new): # commit data to self
        # train
        r = requests.get('http://api.openweathermap.org/data/2.5/weather?q=Cambridge,us&appid=e763734fadaae0d6e5efd2faef74dcf0').json()
        X_new = [r['main']['temperature'], r['main']['humidity'], r['main']['pressure'], r['wind']['speed']]
        X_new, Y_new = np.float_(X_new), np.float_(Y_new)
        self.X = np.append(self.X, X_new, axis=0)
        self.Y = np.append(self.Y, Y_new, axis=0)

    def train(self):
        # train
        X_train = np.copy(self.X)
        Y_train = np.copy(self.Y)
        X_train = (X_train - self.X_scaler[0]) / self.X_scaler[1]
        self.model.fit(X_train, Y_train, epochs=30, verbose=0)

    def predict(self):
        #
        r = requests.get('http://api.openweathermap.org/data/2.5/forecast?q=Cambridge,us&appid=e763734fadaae0d6e5efd2faef74dcf0').json()['list']
        X_pred_list = []
        time_list = []
        for item in r:
            time_list.append(dateparser(item['dt_txt']))
            X_pred_list.append([item['main']['temp'], item['main']['humidity'], item['main']['pressure'], item['wind']['speed']])
        X_pred_list = np.float_(X_pred_list)
        Y_pred_list = []
        # predict
        for idx in range(X_pred_list.shape[0]):
            X_pred = X_pred_list[i]
            X_pred = (X_pred - self.X_scaler[0]) / self.X_scaler[1]
            Y_pred = self.model.predict(X_pred)
            Y_pred_list.append(Y_pred)
        # reverse scale and print
        legends = ['秋衣','衬衣','保暖内衣','毛衣']
        for i in range(Y_pred_list.shape[1]):
            plt.plot(time_list, Y_pred_list[:,i], label=legends[i])
        plt.legend()
        plt.show()
