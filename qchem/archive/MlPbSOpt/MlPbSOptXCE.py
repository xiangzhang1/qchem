# MlPbSOpt, X-CE-version
# ==============================================================================

def V(x):
    return Variable(torch.FloatTensor(np.array(x)), requires_grad=True).cuda()
def C(x):
    return Variable(torch.FloatTensor(np.array(x)), requires_grad=False).cuda()

class MlPbSOptXCE(object):

    def __init__(self):
        # data
        self._X1 = []
        self._y0 = []
        # pipeline
        self.X1_pipeline = StandardScaler()
        self.y_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('10', FunctionTransformer(func=lambda x: x * 15, inverse_func=lambda x: x / 15))
        ])
        # ann
        dropout_p = 0.05
        # self.ce1 = udf_nn(4, 80, 30, 3).cuda()
        self.ce1 = Sequential(
            nn.Linear(4, 80),
            nn.ReLU(),
            nn.Linear(80, 30),
            nn.ReLU(),
            nn.Linear(30, 3)
        ).cuda()

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

    def train(self, n_epochs=200, learning_rate=1E-6, optimizer_name='RMSprop', loss_name='MSELoss', test_set_size=10, total_set_size=None):
    # pipeline
        total_set_size = total_set_size if total_set_size else len(self._y0)
        test_idx = np.random.choice(range(total_set_size), size=test_set_size, replace=False)
        train_idx = np.array([i for i in range(total_set_size) if i not in test_idx])
        _X1 = np.array(self._X1)[train_idx]
        _y0 = np.array(self._y0)[train_idx]
        self.X1_pipeline.fit(np.concatenate(_X1, axis=0))
        _X1 = np.array([self.X1_pipeline.transform(_X1_) for _X1_ in _X1])
        _y0 = self.y_pipeline.fit_transform(_y0)
        ce1 = self.ce1
        # batch
        # ann
        criterion = getattr(nn, loss_name)()
        params = list(ce1.parameters())
        optimizer = getattr(optim, optimizer_name)(params, lr=learning_rate)
        # train
        ce1.train()
        indices = tqdm(np.random.randint(low=0, high=len(_X1), size=n_epochs * len(_X1)))
        for i in indices:
            X1 = V(_X1[i])
            f0 = C(_y0[i])
            f = torch.sum(ce1(X1), keepdim=False, dim=0)
            _f = f.data.cpu().numpy()
            _f0 = f0.data.cpu().numpy()

            loss = criterion(f, f0)
            optimizer.zero_grad()   # suggested trick
            loss.backward()
            optimizer.step()

            if np.random.rand() < 1.0 / 200:
                invy = np.array(self.y_pipeline.inverse_transform(_f.reshape(1,-1))[0])
                invy0 = np.array(self.y_pipeline.inverse_transform(_f0.reshape(1,-1))[0])
                _loss = np.asscalar(loss.data.cpu().numpy())
                indices.set_description('y %s, y0 %s, invy %s, invy0 %s, loss %.2f' %(_f, _f0, invy, invy0, _loss))

        print 'training complete! fuck with the data.'
        IPython.embed()

    def parse_predict(self, cell):
        return self.parse_X1(cell)

    def predict(self, _X1):
        # pipeline
        _X1 = self.X1_pipeline.transform(_X1)
        ce1 = self.ce1
        # ann
        ce1.eval()
        X1 = V(_X1)

        f = torch.sum(ce1(X1), dim=0, keepdim=False)

        # reverse pipeline
        return list(self.y_pipeline.inverse_transform(f.data.cpu().numpy().reshape(1,-1)).reshape(-1))
