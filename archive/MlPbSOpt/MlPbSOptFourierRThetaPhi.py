# MlPbSOptFourier
# =========================

class MlPbSOptFourier(object):

    def __init__(self):
        # data
        self._X = []
        self._y0 = []
        # pipeline
        # ann
        self.model = linear_model.LinearRegression()

    def parse_train(self, bnode, enode):
        bcoor = bnode.cell.ccoor
        ecoor = enode.vasp.optimized_cell.ccoor
        natom0 = bnode.cell.stoichiometry.values()[0]
        ZERO = 1E-6

        for i, bc in enumerate(bcoor):
            rcoor = bcoor - bc
            r = np.linalg.norm(rcoor, axis=1)
            theta = np.arccos(rcoor[:,2] / (r+ZERO))
            sgn = np.sign((i - natom0 + 0.5) * (np.arange(len(rcoor)) - natom0 + 0.5))
            phi = np.arctan2(rcoor[:,1], rcoor[:,0]+ZERO)
            X = np.stack((r * sgn, theta, phi), axis=1)
            X = np.delete(X, i, axis=0)
            self._X.append(X)

            dc = (ecoor - bcoor)[i]
            r = np.linalg.norm(dc)
            theta = np.arccos(dc[0] / (r+ZERO))
            phi = np.arctan2(dc[1], dc[0]+ZERO)
            self._y0.append([r, theta, phi])

    def train(self):
        X = self._X
        for i in range(len(X)):
            X[i][:,0] = X[i][:,0] ** -1.0
        X_scaler = StandardScaler()
        X_scaler.fit([x for scoor in X for x in scoor])
        X = [X_scaler.transform(X_) for X_ in X]

        y_scaler = StandardScaler()
        y0 = y_scaler.fit_transform(np.array(self._y0))

        X_processed = []
        y0_processed = []
        for X_, y0_ in zip(X, y0):

            Xi0i1i2i3 = np.zeros((9,9,9))
            for i0, i1, i2 in itertools.permutations(np.arange(-4, 5), 3):
                def f(i):
                    if i<0:
                        return lambda x:np.cos(i * x)
                    if i>0:
                        return lambda x:np.sin(i * x)
                    if i==0:
                        return lambda x:1
                f0, f1, f2 = f(i0), f(i1), f(i2)
                Xi0i1i2i3[i0+2, i1+2, i2+2] = np.sum(f0(i0 * X_[:, 0]) * f1(i1 * X_[:, 1]) * f2(i2 * X_[:, 2]))

            X_processed.append(Xi0i1i2i3.flatten())
            y0_processed.append(y0_[0])

        self.model.fit(X_processed, y0_processed)

        # evaluate
        y_processed = self.model.predict(X_processed)
        IPython.embed()

    def parse_predict(self, cell):
        return self.parse_X(cell)

    def predict(self, _X):
        return self.model.predict(_X)
