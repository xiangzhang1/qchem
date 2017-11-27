# MlPbSOptr
# ==============================================================================


class MlPbSOptScaler(BaseEstimator,TransformerMixin):       # THIS IS NOT USEFUL AT ALL! DOES NOT SCALE DIFFERENT DIMENSIONS.
    def __init__(self):
        self.mean = 0

    def fit(self, X):
        self.mean = np.mean([np.mean(np.abs(subX)) for subX in X])
        return self

    def transform(self, X):
        return X / self.mean / 1.7

    def inverse_transform(self, X):
        return X * self.mean * 1.7


class MlPbSOpt(object):
    '''Deprecated.'''

    def __init__(self):
        # data
        self._X = []
        self._y0 = []
        # pipeline
        self.X_pipeline = MlPbSOptScaler()
        self.y_pipeline = Pipeline([
            ('scaler', MlPbSOptScaler()),
            ('10', FunctionTransformer(func=lambda x: x * 5, inverse_func=lambda x: x / 5))
        ])
        # ann
        self.nets = {-2: udf_nn(1,5,10,5,1), 0:udf_nn(1,5,10,5,1), 2:udf_nn(1,5,10,5,1)}

    def parse_train(self, vasp):
        # checks
        gen = vasp.node().gen
        vasprunxml_lastline = vasp.ssh_and_run('tail -1 %s/vasprun.xml' %vasp.remote_folder_name).splitlines()[0]
        if not (gen.parse_if('opt') and vasp.info('n_ionic_step') < int(gen.getkw('nsw')) and '</modeling>' in vasprunxml_lastline):
            raise shared.CustomError('{}.compute: not optimization cell, or terminated prematurely. skipped :)'.format(self.__class__.__name__))

        a = 6.01417/2
        cell = vasp.optimized_cell

        ccoor = np.copy(cell.ccoor)
        origin = ccoor[cell.ccoor_kdtree().query(np.mean(ccoor, axis=0))[1]]    # closest-to-center atom
        def error_after_transformation(origin, ccoor=ccoor, a=a):       # snap to grid. note: parallelization doesn't save time.
            fcoor = (ccoor - origin) / a
            return np.sum(np.abs(fcoor - np.around(fcoor)))
        origin = minimize(fun=error_after_transformation,
                     x0=origin,
                     bounds=[(origin[i]-0.2*a, origin[i]+0.2*a) for i in range(3)],
                     tol=1e-10
                    ).x
        ccoor = ccoor - origin + np.around(origin / a) * a      # on-grid coordinate. vaguely resemble the original cell

        # parse and store
        pbs_order_factor = 1 if cell.stoichiometry.keys()[0]=='Pb' else -1
        for idx_atom, c in enumerate(ccoor):
            relative_ccoor = ccoor - c
            coor_sgn = np.sign(np.arange(ccoor.shape[0]) - cell.stoichiometry.values()[0] + 0.5) * pbs_order_factor
            sgn = np.sign(idx_atom - cell.stoichiometry.values()[0] + 0.5) * pbs_order_factor
            feature = np.concatenate((relative_ccoor, np.c_[[sgn] * ccoor.shape[0]], np.c_[coor_sgn]), axis=1)
            feature = np.delete(feature, idx_atom, 0)
            label = c - np.around(c/a)*a
            self._X.append(feature)
            self._y0.append(label)



    def train(self, n_epochs=400, learning_rate=0.001, optimizer_name='Adam'):
        # pipeline
        _X = copy.deepcopy(self._X)
        self.X_pipeline.fit(np.concatenate([_subX[:,:3] for _subX in _X], axis=0))
        for i in range(len(_X)):
            _X[i][:,:3] = self.X_pipeline.transform(_X[i][:,:3])
        _y0 = self.y_pipeline.fit_transform(self._y0)
        # batch
        # ann
        criterion = nn.MSELoss()
        params = list(self.nets[-2].parameters()) + list(self.nets[0].parameters()) + list(self.nets[2].parameters())
        optimizer = getattr(optim, optimizer_name)(params, lr=learning_rate)
        # train
        [net.train() for net in self.nets.values()]
        for epoch in range(n_epochs):
            for _X_batch, _y0_batch in zip(_X[:-50], _y0[:-50]):
                dx = Variable(torch.zeros(3))
                for sgn in [-2,0,2]:
                    indices, = np.where([row[3] + row[4] == sgn for row in _X_batch])
                    if not len(indices): continue
                    X = Variable(torch.FloatTensor(_X_batch[indices]))

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
        print self.__class__.__name__ + '.train: training finished. evaluation on last items: \n actual | error'
        for i in range(len(self._X)-50, len(self._X)):
            _X = np.array(self._X)[i]
            _y0 = np.float32(self._y0)[i]
            _y = np.float32(self.predict(_X))[0]
            print _y0, _y

    def parse_predict(self, cell): # cell -> (Natom, xyzs)
        a = 6.01417/2
        ccoor = cell.ccoor
        # parse and store
        features = []
        pbs_order_factor = 1 if cell.stoichiometry.keys()[0]=='Pb' else -1
        for idx_atom, c in enumerate(ccoor):
            relative_ccoor = ccoor - c
            coor_sgn = np.sign(np.arange(ccoor.shape[0]) - cell.stoichiometry.values()[0] + 0.5) * pbs_order_factor
            sgn = np.sign(idx_atom - cell.stoichiometry.values()[0] + 0.5) * pbs_order_factor
            feature = np.concatenate((relative_ccoor, np.c_[[sgn] * ccoor.shape[0]], np.c_[coor_sgn]), axis=1)
            feature = np.delete(feature, idx_atom, 0)
            features.append(feature)
        return features

    def predict(self, _X):  # (Natom, xyzs) -> (1, Natom)
        # pipeline
        _X = copy.deepcopy(_X)
        _X[:,:3] = self.X_pipeline.transform(_X[:,:3])
        # ann
        [net.eval() for net in self.nets.values()]

        dx = Variable(torch.zeros(3))
        for sgn in [-2,0,2]:
            indices, = np.where([row[3] + row[4] == sgn for row in _X])
            if not len(indices): continue
            X = Variable(torch.FloatTensor(_X[indices]))

            r = torch.norm(X[:, :3], p=2, dim=1, keepdim=True)      # (N,3) -> (N,1)
            rhat = X[:, :3] / r * X[:, 3:4] * X[:, 4:5]     # (N,3) / (N,1)
            dxi = self.nets[sgn](r) * rhat     # (N,1) * (N,1) * (N,1) * (N,3)
            dx += torch.sum(dxi, dim=0, keepdim=False)    # (N,3) -> (3)

        # pipeline
        _y_inverse = self.y_pipeline.inverse_transform(dx.data.numpy().reshape(1,-1)).reshape(-1)   # Scaler requries 2D array.
        return _y_inverse


    def optimize(self, cell):
        #
        cell = copy.deepcopy(cell)
        Xs = self.parse_predict(cell)
        #
        for idx_atom in range(cell.natoms()):
            X = Xs[idx_atom]
            dx = self.predict(X)
            cell.ccoor[idx_atom] += dx
        return cell
