# MlPbSOpt 级联 第二层
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class MlPbSOptL2(object):

    def __init__(self):
        # data
        self._X = []
        self._y0 = []
        # scaler
        self.X_pipeline = StandardScaler()
        self.y_pipeline = StandardScaler()
        # ANN(deep)
        self.net = Sequential(
            nn.Conv3d(2, 4, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(4, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool3d(2),
            Reshape(-1, 64),
            nn.Linear(64, 6),
            nn.ELU(),
            nn.Linear(6, 6),
            nn.ELU(),
            nn.Linear(6, 3)
        )

    def train(self, n_epochs=1000, batch_size=64, learning_rate=0.001, optimizer_name='Adam', test_set_size=128):
        test_idx = np.random.choice(range(len(self._X)), size=test_set_size)
        train_idx = np.array([i for i in range(len(self._X)) if i not in test_idx])

        # train
        # pipeline
        _X = np.array(self._X)
        shape = _X.shape
        _X = self.X_pipeline.fit_transform(_X.reshape(shape[0], -1)).reshape(_X.shape)[train_idx]
        _y0 = self.y_pipeline.fit_transform(self._y0)[train_idx]
        # batch: random.choice
        # ann
        criterion = nn.MSELoss()
        optimizer = getattr(optim, optimizer_name)(self.net.parameters(), lr=learning_rate)
        # train
        self.net.train()
        for epoch in range(n_epochs):
            batch_idx = np.random.choice(range(_X.shape[0]), size=batch_size)
            X_batch= Variable(torch.FloatTensor(_X[batch_idx]), requires_grad=True)
            y0_batch = Variable(torch.FloatTensor(_y0[batch_idx]), requires_grad=False)
            y = self.net(X_batch)
            loss = criterion(y, y0_batch)
            optimizer.zero_grad()   # suggested trick
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print 'epoch %s, loss %s'%(epoch, loss.data.numpy()[0])

        # test
        _X = np.array(self._X)[test_idx]
        _y0 = np.float32(self._y0)[test_idx]
        _y = np.float32(self.predict(_X))
        print self.__class__.__name__ + '.train: training finished. evaluation on last items: \n actual | predicted'
        for a, b in zip(_y0, _y-_y0):
            print np.linalg.norm(a), np.linalg.norm(b)

    def parse_train(self, X1, y1):  # 1 bunch in many out
        X1 = np.float32(X1) ; y1 = np.float32(y1)
        a = 6.01417/2
        for i in range(len(X1)):
            X1_, y1_ = X1[i], y1[i]
            #
            X2_ = np.zeros((2,11,11,11))    # 2 channels
            for c in X1_:
                rc = np.int32(np.around(c[:3] / a)) + 5
                if all(-1<x<11 for x in rc):
                    X2_[0,rc[0],rc[1],rc[2]] = c[3] * c[4]
                    X2_[1,rc[0],rc[1],rc[2]] = 1
            self._X.append(X2_)
            # y2_ = y1_
            self._y0.append(y1_)


    def predict(self, _X):  # many in many out
        X1 = np.float32(X1) ; y1 = np.float32(y1)
        # pipeline
        _X = np.array(_X)
        shape = _X.shape
        _X = self.X_pipeline.transform(_X.reshape(shape[0], -1)).reshape(_X.shape)
        # ann
        self.net.eval()
        y = self.net(Variable(torch.FloatTensor(_X), requires_grad=True))
        # pipeline
        _y_inverse = self.y_pipeline.inverse_transform(y.data.numpy())
        return _y_inverse


    def parse_predict(self, X1_):   # many in many out
        a = 6.01417/2
        X2 = []
        for i in range(len(X1)):
            X1_, y1_ = X1[i], y1[i]
            #
            X2_ = np.zeros((2,11,11,11))    # 2 channels
            for c in X1_:
                rc = np.int32(np.around(c[:3] / a)) + 5
                if all(-1<x<11 for x in rc):
                    X2_[0,rc[0],rc[1],rc[2]] = c[3] * c[4]
                    X2_[1,rc[0],rc[1],rc[2]] = 1
            X2._X.append(X2_)
        return X2
