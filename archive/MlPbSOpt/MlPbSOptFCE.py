def V(x):
    return Variable(torch.FloatTensor(np.array(x)), requires_grad=True)
def C(x):
    return Variable(torch.FloatTensor(np.array(x)), requires_grad=False)

class MlPbSOptFCE(object):

    def __init__(self):
        # data
        self._X1 = []
        self._y0 = []
        # pipeline
        # self.X1_pipeline = StandardScaler(with_mean=False)
        # self.y_pipeline = StandardScaler(with_mean=False)
        # ann
        self.ce1 = udf_nn(4, 400, 20, 1)

    def parse_X1(self, ccoor, natom0):
        '''
        ccoor is coordinates. natom0 is # of Pb atoms (for determining sgn).
        returns a list.
        '''
        X1 = []
        for i, c in enumerate(ccoor):
            dcoor = ccoor - c
            sgn = np.sign((i - natom0 + 0.5) * (np.arange(len(ccoor)) - natom0 + 0.5))
            dcoorp = np.concatenate((dcoor, np.c_[sgn]), axis=1)
            X1.append(np.delete(dcoorp, i, axis=0))
        return X1

    def parse_y0(self, path, natom):
        os.chdir(path)
        with open('OUTCAR', 'r') as f:
            lines = f.readlines()
        i = next(i for i, line in enumerate(lines) if 'TOTAL-FORCE' in line)
        force_lines = lines[i+2: i+2+natom]
        forces = np.float32([line.split() for line in force_lines])[:, 3:]
        return forces

    def parse_train(self, vasp):
        '''More of a handle.'''
        cell = vasp.optimized_cell if getattr(vasp, 'optimized_cell', None) else vasp.node().cell
        natom0 = cell.stoichiometry.values()[0]
        natom = sum(cell.stoichiometry.values())
        path = vasp.node().path
        self._X1 += list(self.parse_X1(cell.ccoor, natom0))
        self._y0 += list(self.parse_y0(path, natom))

    def train(self, n_epochs=10, learning_rate=0.01, optimizer_name='Adam'):
        # pipeline
        # self.X1_pipeline.fit(np.concatenate(self._X1, axis=0))
        # _X1 = np.array([self.X1_pipeline.transform(_X1_) for _X1_ in self._X1])
        # _y0 = self.y_pipeline.fit_transform(self._y0)
        _X1 = self._X1
        _y0 = self._y0
        ce1 = self.ce1
        # batch
        # ann
        criterion = nn.MSELoss()
        params = list(ce1.parameters())
        optimizer = getattr(optim, optimizer_name)(params, lr=learning_rate)
        # train
        ce1.train()
        t = trange(n_epochs * len(_X1), desc='loss: 0')
        for epoch in t:
            i = np.random.randint(0, len(_X1) - 50)
            X1 = V(_X1[i])
            f0 = C(_y0[i])

            X1m = X1.clone()
            origin = V([0,0,0,0])
            X1m = X1 - origin
            e = torch.sum(ce1(X1m), keepdim=False)
            f = torch.autograd.grad(e, origin, create_graph=True)[0][0:3]

            loss = criterion(f, f0)
            optimizer.zero_grad()   # suggested trick
            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                t.set_description('loss: %s, f: %s, f0: %s' %(np.asscalar(loss.data.numpy()), f.data.numpy(), f0.data.numpy()))

    def parse_predict(self, ccoor, natom0):
        return self.parse_X1(ccoor, natom0)

    def predict_e(self, _X1):
        # pipeline
        ce1 = self.ce1
        # ann
        ce1.eval()
        y = torch.sum(ce1(V(_X1)), keepdim=True)
        # reverse pipeline
        return y.data.numpy()

    def predict_f(self, _X1):
        # pipeline
        ce1 = self.ce1
        # ann
        ce1.eval()
        X1 = V(_X1)

        X1m = X1.clone()
        origin = V([0,0,0])
        X1m[:,:3] = X1[:,:3] - origin
        e = torch.sum(ce1(X1m), keepdim=False)
        f = torch.autograd.grad(e, origin, create_graph=True)[0]

        # reverse pipeline
        return f.data.numpy()
