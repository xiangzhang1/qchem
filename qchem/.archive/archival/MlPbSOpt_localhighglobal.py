# MlPbSOpt
# ==============================================================================

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class MlPbSOpt(object):

    def __init__(self):
        # data
        self._X_local = []
        self._X_high = []
        self._X_global = []
        self._y0 = []
        # pipeline
        self.X_high_pipeline = StandardScaler()
        self.y_pipeline = StandardScaler()
        # ann. have fun!
        bn_momentum = 0.74
        dropout_p = 0.001
        self.net_local = Sequential(
            nn.Linear(125, 25),
            nn.ELU(),
            nn.BatchNorm1d(25, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(25, 20),
            nn.ELU(),
            nn.BatchNorm1d(20, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(20, 15),
            nn.ELU(),
            nn.BatchNorm1d(15, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(15, 10),
            nn.ELU(),
            nn.BatchNorm1d(10, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(10, 5),
            nn.ELU(),
            nn.BatchNorm1d(5, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
        ).cuda()
        self.net_high = Sequential(
            nn.Linear(12, 8),
            nn.ELU(),
            nn.BatchNorm1d(8, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(8, 7),
            nn.ELU(),
            nn.BatchNorm1d(7, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(7, 6),
            nn.ELU(),
            nn.BatchNorm1d(6, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
        ).cuda()
        self.net_global = Sequential(
            nn.Conv3d(1, 4, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(4, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool3d(2),
            Reshape(-1, 8 * 2 * 2 * 2),
            nn.Linear(8 * 2 * 2 * 2, 16),
            nn.ELU(),
            nn.BatchNorm1d(16, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(16, 12),
            nn.ELU(),
            nn.BatchNorm1d(12, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(12, 8),
            nn.ELU(),
            nn.BatchNorm1d(8, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(8, 4),
            nn.ELU(),
            nn.BatchNorm1d(4, momentum=bn_momentum),
            nn.Dropout(p=dropout_p)
        ).cuda()
        self.net_final = Sequential(
            nn.Linear(15, 12),
            nn.ELU(),
            nn.BatchNorm1d(12, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(12, 8),
            nn.ELU(),
            nn.BatchNorm1d(8, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(8, 4),
            nn.ELU(),
            nn.BatchNorm1d(4, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(4, 1)
        ).cuda()


    def parse_train(self, vasp):
        a = 6.01417/2
        cell = vasp.optimized_cell

        # 一.
        ccoor = cell.ccoor
        origin = ccoor[cell.ccoor_kdtree().query(np.mean(ccoor, axis=0))[1]]
        def error_after_transformation(origin, ccoor=ccoor, a=a):       # note: parallelization doesn't save time.
            fcoor = (ccoor - origin) / a
            return np.sum(np.abs(fcoor - np.around(fcoor)))
        origin = minimize(fun=error_after_transformation,
                     x0=origin,
                     bounds=[(origin[i]-0.2*a, origin[i]+0.2*a) for i in range(3)],
                     tol=1e-10
                    ).x
        ccoor_transformed = ccoor - origin + np.around(origin / a) * a + 5 * a
        fcoor = ccoor_transformed / a
        rfcoor = np.int32(np.around(fcoor))

        # 二.
        Nx = int(math.ceil(np.linalg.norm(cell.base[0]) / a)) + 10      # assume 长方体
        Ny = int(math.ceil(np.linalg.norm(cell.base[1]) / a)) + 10
        Nz = int(math.ceil(np.linalg.norm(cell.base[2]) / a)) + 10
        dense_matrix = np.zeros((Nx, Ny, Nz, 4))
        for idx_atom, rfc in enumerate(rfcoor):
            # 坐标
            ix, iy, iz = rfc
            # 符号位
            idx_ele = 0 if idx_atom < cell.stoichiometry.values()[0] else 1     # OrderedDict顺序     # assume Pb S only
            ele = cell.stoichiometry.keys()[idx_ele]
            dense_matrix[ix, iy, iz, 0] = 1 if ele=='Pb' else -1
            # 数值位: dx, dy, dz
            dense_matrix[ix, iy, iz, 1:4] = fcoor[idx_atom] - rfc

        feature_stoichiometry = np.float32([cell.stoichiometry['Pb'] - cell.stoichiometry['S'], cell.natoms() / 100.0])
        rfcenter = np.mean(rfcoor, axis=0)

        for rfc in rfcoor:
            ix, iy, iz = rfc

            # 三二. 取得 dx 和 local feature
            feature_local = dense_matrix[ix-2:ix+3, iy-2:iy+3, iz-2:iz+3, 0].flatten() * dense_matrix[ix, iy, iz, 0]  #27
            label_dx = dense_matrix[ix, iy, iz, 1]  #dx

            feature_global = np.abs(dense_matrix[ix-6:ix+7, iy-6:iy+7, iz-6:iz+7, 0]) #shape=(13,13,13)

            # 四. 关于高级策略
            feature_selfcharge = dense_matrix[ix, iy, iz, 0]  #scalar
            feature_displace_to_center = np.float32([ix,iy,iz]) - rfcenter

            nsdx1 = next(k for k,g in enumerate(dense_matrix[ix:,iy,iz,0]) if g==0)
            nsdx2 = next(k for k,g in enumerate(dense_matrix[ix:0:-1,iy,iz,0]) if g==0)
            nsdy1 = next(k for k,g in enumerate(dense_matrix[ix,iy:,iz,0]) if g==0)
            nsdy2 = next(k for k,g in enumerate(dense_matrix[ix,iy:0:-1,iz,0]) if g==0)
            nsdz1 = next(k for k,g in enumerate(dense_matrix[ix,iy,iz:,0]) if g==0)
            nsdz2 = next(k for k,g in enumerate(dense_matrix[ix,iy,iz:0:-1,0]) if g==0)

            self._X_local.append(feature_local)
            self._X_high.append(np.concatenate((feature_stoichiometry, [feature_selfcharge], feature_displace_to_center, [nsdx1, nsdx2, nsdy1, nsdy2, nsdz1, nsdz2])))
            self._X_global.append([feature_global])
            self._y0.append([label_dx])


    def train(self, n_epochs=8000, batch_size=64, learning_rate=0.001, optimizer_name='SGD', test_set_size=128):
        test_idx = np.random.choice(range(len(self._X_local)), size=test_set_size)
        train_idx = np.array([i for i in range(len(self._X_local)) if i not in test_idx])

        # train
        # pipeline
        _X_local = np.float32(self._X_local)[train_idx]
        _X_high = self.X_high_pipeline.fit_transform(self._X_high)[train_idx]
        _X_global = np.float32(self._X_global)[train_idx]
        _y0 = self.y_pipeline.fit_transform(self._y0)[train_idx]
        # batch: random.choice
        # ann
        criterion = nn.MSELoss()
        optimizer = getattr(optim, optimizer_name)(itertools.chain(self.net_local.parameters(), self.net_high.parameters(), self.net_global.parameters(), self.net_final.parameters()), lr=learning_rate)
        # train
        self.net_local.train()
        self.net_high.train()
        self.net_global.train()
        self.net_final.train()
        for epoch in range(n_epochs):
            batch_idx = np.random.choice(range(_X_local.shape[0]), size=batch_size)
            #
            X_local= Variable(torch.FloatTensor(_X_local[batch_idx]), requires_grad=True).cuda()
            X_high= Variable(torch.FloatTensor(_X_high[batch_idx]), requires_grad=True).cuda()
            X_global= Variable(torch.FloatTensor(_X_global[batch_idx]), requires_grad=True).cuda()
            y0 = Variable(torch.FloatTensor(_y0[batch_idx]), requires_grad=False).cuda()
            #
            y_local = self.net_local(X_local)
            y_high = self.net_high(X_high)
            y_global = self.net_global(X_global)
            X_final = torch.cat((y_local, y_high, y_global), dim=1)
            y = self.net_final(X_final)
            #
            loss = criterion(y, y0)
            optimizer.zero_grad()   # suggested trick
            loss.backward()
            optimizer.step()
            if epoch % 256 == 0:
                print 'epoch %s, loss %s'%(epoch, loss.data.cpu().numpy()[0])

        # test
        _X_local = np.array(self._X_local)[test_idx]
        _X_high = np.array(self._X_high)[test_idx]
        _X_global = np.array(self._X_global)[test_idx]
        _y0_flat = np.float32(self._y0).flatten()[test_idx]
        _y_flat = np.float32(self.predict(_X_local, _X_high, _X_global)).flatten()
        print self.__class__.__name__ + '.train: training finished. evaluation on last items: \n actual | predicted'
        for a, b in zip(_y0_flat, _y_flat):
            print a, b


    def parse_predict(self, gen, cell, makeparam):
        pass  # 未完待续

    def predict(self, _X_local, _X_high, _X_global):
        # pipeline
        _X_high = self.X_high_pipeline.transform(_X_high)
        # ann
        self.net_local.eval()
        self.net_high.eval()
        self.net_global.eval()
        self.net_final.eval()
        X_local= Variable(torch.FloatTensor(_X_local), requires_grad=True).cuda()
        X_high= Variable(torch.FloatTensor(_X_high), requires_grad=True).cuda()
        X_global= Variable(torch.FloatTensor(_X_global), requires_grad=True).cuda()
        #
        y_local = self.net_local(X_local)
        y_high = self.net_high(X_high)
        y_global = self.net_global(X_global)
        X_final = torch.cat((y_local, y_high, y_global), dim=1)
        y = self.net_final(X_final)
        # pipeline
        _y_inverse = self.y_pipeline.inverse_transform(y.data.cpu().numpy())
        return _y_inverse
