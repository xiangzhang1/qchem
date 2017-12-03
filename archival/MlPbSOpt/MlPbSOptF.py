# MlPbSOptF
# ==============================================================================


class MlPbSOptF(object):

    def __init__(self):
        # data
        self._X = []
        self._y0 = []

        # pipeline
        # method 1
        self.X_pipeline = MlPbSOptScaler()
        self.y_pipeline = Pipeline([
            ('scaler', MlPbSOptScaler()),
            ('10', FunctionTransformer(func=lambda x: x * 10, inverse_func=lambda x: x / 10))
        ])
        # # method 2
        # self.X_pipeline = StandardScaler()
        # self.y_pipeline = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('_10', FunctionTransformer(func=lambda x: x * 10, inverse_func=lambda x: x / 10))
        # ])

        # ann
        self.nets = {-2: udf_nn(1,5,10,5,1), 0:udf_nn(1,5,10,5,1), 2:udf_nn(1,5,10,5,1)}

    def parse_train(self, vasp):
        a = 6.01417/2
        cell = vasp.optimized_cell
        ccoor = cell.ccoor

        # parse and store
        os.chdir(vasp.node().path)
        with open('OUTCAR', 'r') as f:
            lines = f.readlines()
        i = next(i for i, line in enumerate(lines) if 'TOTAL-FORCE' in line)
        force_lines = lines[i+2: i+2+cell.natoms()]
        forces = np.float32([line.split() for line in force_lines])[:, 3:]


        pbs_order_factor = 1 if cell.stoichiometry.keys()[0]=='Pb' else -1
        for idx_atom, c in enumerate(ccoor):
            relative_ccoor = ccoor - c
            coor_sgn = np.sign(np.arange(ccoor.shape[0]) - cell.stoichiometry.values()[0] + 0.5) * pbs_order_factor
            sgn = np.sign(idx_atom - cell.stoichiometry.values()[0] + 0.5) * pbs_order_factor
            feature = np.concatenate((relative_ccoor, np.c_[[sgn] * ccoor.shape[0]], np.c_[coor_sgn]), axis=1)
            feature = np.delete(feature, idx_atom, 0)
            self._X.append(feature)
            self._y0.append(forces[idx_atom])


    def train(self, n_epochs=200, learning_rate=0.001, optimizer_name='Adam'):

        # train
        # pipeline
        _X = copy.deepcopy(self._X)
        self.X_pipeline.fit(np.concatenate([_subX[:,:3] for _subX in _X], axis=0))
        for i in range(len(_X)):
            _X[i][:,:3] = self.X_pipeline.transform(_X[i][:,:3])
        _y0 = self.y_pipeline.fit_transform(self._y0)
        # batch: random.choice
        # ann
        criterion = nn.MSELoss()
        params = list(self.nets[-2].parameters()) + list(self.nets[0].parameters()) + list(self.nets[2].parameters())
        optimizer = getattr(optim, optimizer_name)(params, lr=learning_rate)
        # train
        [net.train() for net in self.nets.values()]
        for epoch in range(n_epochs):
            for _X_batch, _y0_batch in zip(_X[:-50], _y0[:-50]):

                # method 3
                dx = Variable(torch.zeros(3))
                for sgn in [-2,0,2]:
                    indices, = np.where([row[3] + row[4] == sgn for row in _X_batch])
                    if not len(indices): continue
                    X = Variable(torch.FloatTensor(_X_batch[indices]))

                    # # method 1
                    # r = torch.norm(X[:, :3], p=2, dim=1, keepdim=True)      # (N,3) -> (N,1)
                    # rhat = X[:, :3] / r     # (N,3) / (N,1)
                    # dx = self.net(r) * X[:, 3:4] * X[:, 4:5] * rhat     # (N,1) * (N,1) * (N,1) * (N,3)
                    # dx = torch.sum(dx, dim=0, keepdim=False)    # (N,3) -> (3)
                    # # method 2
                    # dx = self.net(X)    #(N,3) * (N,1) * (N,1)
                    # dx = torch.sum(dx, dim=0, keepdim=False)    # (N,3) -> (3)
                    # method 3
                    r = torch.norm(X[:, :3], p=2, dim=1, keepdim=True)      # (N,3) -> (N,1)
                    rhat = X[:, :3] / r * X[:, 3:4] * X[:, 4:5]     # (N,3) / (N,1)
                    dxi = self.nets[sgn](r) * rhat     # (N,1) * (N,1) * (N,1) * (N,3)
                    dx += torch.sum(dxi, dim=0, keepdim=False)    # (N,3) -> (3)

                dx0 = Variable(torch.FloatTensor(_y0_batch))
                loss = criterion(dx, dx0)
                optimizer.zero_grad()   # suggested trick
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print 'epoch %s, loss %s' %(epoch, np.asscalar(loss.data.numpy()))

        # test
        print self.__class__.__name__ + '.train: training finished. evaluation on last items: \n actual | predicted'

        for i in range(len(self._X)-50, len(self._X)):
            _X = np.array(self._X)[i]
            _y0 = np.float32(self._y0)[i]
            _y = np.float32(self.predict(_X))[0]
            print _y0, _y-_y0


    def parse_predict(self, ccoor):  # 1cell-in-many-out. note: on-grid input assumed!
        a = 6.01417/2

        # parse and store
        features = []
        for idx_atom, c in enumerate(ccoor):
            relative_ccoor = ccoor - c
            coor_sgn = np.sign(np.arange(ccoor.shape[0]) - 55 + 0.5)
            sgn = np.sign(idx_atom - 55 + 0.5)
            feature = np.concatenate((relative_ccoor, np.c_[[sgn] * ccoor.shape[0]], np.c_[coor_sgn]), axis=1)
            feature = np.delete(feature, idx_atom, 0)
            features.append(feature)

        return features


    def predict(self, _X):  # EXCEPTION: 1-in 1-out. This special ANN doesn't take many-in-many-out
        # pipeline
        _X = copy.deepcopy(_X)
        _X[:,:3] = self.X_pipeline.transform(_X[:,:3])
        # ann
        [net.eval() for net in self.nets.values()]

        # # method 1
        # X = Variable(torch.FloatTensor(_X))
        # r = torch.norm(X[:, :3], p=2, dim=1, keepdim=True)      # (N,3) -> (N,1)
        # rhat = X[:, :3] / r     # (N,3) / (N,1)
        # dx = self.net(r) * X[:, 3:4] * X[:, 4:5] * rhat     # (N,1) * (N,1) * (N,1) * (N,3)
        # dx = torch.sum(dx, dim=0, keepdim=False)    # (N,3) -> (3)
        # # method 2
        # X = Variable(torch.FloatTensor(_X))
        # dx = self.net(X)    #(N,3) * (N,1) * (N,1)
        # dx = torch.sum(dx, dim=0, keepdim=False)    # (N,3) -> (3)

        # method 3
        dx = Variable(torch.zeros(3))
        for sgn in [-2,0,2]:
            indices, = np.where([row[3] + row[4] == sgn for row in _X])
            if not len(indices): continue
            X = Variable(torch.FloatTensor(_X[indices]))

            # # method 1
            # r = torch.norm(X[:, :3], p=2, dim=1, keepdim=True)      # (N,3) -> (N,1)
            # rhat = X[:, :3] / r     # (N,3) / (N,1)
            # dx = self.net(r) * X[:, 3:4] * X[:, 4:5] * rhat     # (N,1) * (N,1) * (N,1) * (N,3)
            # dx = torch.sum(dx, dim=0, keepdim=False)    # (N,3) -> (3)
            # # method 2
            # dx = self.net(X)    #(N,3) * (N,1) * (N,1)
            # dx = torch.sum(dx, dim=0, keepdim=False)    # (N,3) -> (3)
            # method 3
            r = torch.norm(X[:, :3], p=2, dim=1, keepdim=True)      # (N,3) -> (N,1)
            rhat = X[:, :3] / r * X[:, 3:4] * X[:, 4:5]     # (N,3) / (N,1)
            dxi = self.nets[sgn](r) * rhat     # (N,1) * (N,1) * (N,1) * (N,3)
            dx += torch.sum(dxi, dim=0, keepdim=False)    # (N,3) -> (3)

        # pipeline
        _y_inverse = self.y_pipeline.inverse_transform(dx.data.numpy().reshape(1,-1))
        return _y_inverse


    def optimize_(self, ccoor, speed=0.001, factor=0.6):
        '''
        Optimize in place.
        '''
        a = 6.01417/2
        def cost(ccoor, self=self):
            Xs = self.parse_predict(ccoor.reshape(93,3))
            fs = [self.predict(X) for X in Xs]
            # cell.ccoor[idx_atom] += (speed * f).reshape(3)
            # print '---------------------------'
            # print ccoor
            # print '---------------------------'
            print 'max force is %s eV/A' %(np.amax(np.abs(fs)))
            a = np.random.rand()
            if a < 0.01:
                print '---------------------------'
                print ccoor
                print '---------------------------'
            return np.amax(np.abs(fs))
        ccoor = minimize(fun=cost,
                     method='Powell',
                     x0=ccoor.reshape(-1),
                     tol=1e-4,
                     options={'maxiter':10}
                     # bounds=[(x-0.2*a, x+0.2*a) for x in ccoor.reshape(-1)],
                    ).x
        return ccoor
        # old_maxf = 9
        # while True:
        #     Xs = self.parse_predict(ccoor.reshape(93,3))
        #     fs = np.float32([self.predict(X)[0] for X in Xs])
        #     ccoor += speed * fs
        #     maxf = np.amax(np.abs(fs))
        #     if maxf > old_maxf:
        #         speed *= -1 * factor
        #     old_maxf = maxf
        #     print 'speed %s, max force is %s eV/A' %(speed, maxf)
        # return ccoor
