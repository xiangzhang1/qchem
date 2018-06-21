class Ml_vasp_memory(object):

    def __init__(self):
        self.X_train = np.float_([])
        self.Y_train = np.float_([])
        self.X_scaler = np.float_([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [10**9, 1, 10**9, 10**9, 1, 1000, 1, 1]
        ])
        self.Y_scaler = np.float_([
            [0],
            [10**9]
        ])
        self.model = Sequential([
            Dense(8, activation='relu', input_dim=8),
            Dropout(0.05),
            Dense(4, activation='relu'),
            Dropout(0.05),
            Dense(1, activation='linear')
        ])
        self.model.compile(optimizer='rmsprop',
                      loss='mse')

    @shared.debug_wrap
    def take_data(self, node): # commit data to self
        ZERO = 0.1
        makeparam = Makeparam(node.gen)
        input_ = np.float_([[
                            makeparam.projector_real,
                            makeparam.projector_reciprocal,
                            makeparam.wavefunction,
                            makeparam.arraygrid,
                            node.cell.natoms(),
                            np.dot(np.cross(node.cell.base[0], node.cell.base[1]), node.cell.base[2]),
                            node.gen.getkw('npar'),
                            node.gen.ncore_total()
                         ]])
        label = [node.vasp.memory_used()]
        if not node.vasp.memory_used():
            print self.__class__.__name__ + ': cannot extract memory used. Aborted. '
        if shared.VERBOSE >= 1: print self.__class__.__name__ + ': taking data... ',
        if not any([np.linalg.norm(input_ - row) < ZERO for row in self.X_train]):
            self.X_train = np.append(self.X_train, input_, axis=0)
            self.Y_train = np.append(self.Y_train, label, axis=0)
            if shared.VERBOSE >= 1: print 'Complete.'
        else:
            if shared.VERBOSE >= 1: print 'Skipped.'

    def scale_and_fit(self):
        # scale
        X_train = (self.X_train - self.X_scaler[0]) / self.X_scaler[1]
        Y_train = (self.Y_train - self.Y_scaler[0]) / self.Y_scaler[1]
        # fit
        self.model.fit(X_train, Y_train, epochs=30, verbose=0)

    def scale_and_predict(self, X_test):
        X_test = (X_test - self.X_scaler[0]) / self.X_scaler[1]
        Y_test = self.model.predict(X_test)
        return Y_test * self.Y_scaler[1] + self.Y_scaler[0]

    def make_prediction(self, gen):
        makeparam = Makeparam(gen)
        X_test = np.float_([[
                            makeparam.projector_real,
                            makeparam.projector_reciprocal,
                            makeparam.wavefunction,
                            makeparam.arraygrid,
                            gen.cell.natoms(),
                            np.dot(np.cross(gen.cell.base[0], gen.cell.base[1]), gen.cell.base[2]),
                            gen.getkw('npar'),
                            gen.ncore_total()
                         ]])
        Y_test = self.scale_and_predict(X_test)
        if shared.VERBOSE >= 2:   print 'X_test is %s; Y_test is %s' %(X_test, Y_test)
        return np.asscalar(Y_test)

    def make_prediction2(self, gen):
        makeparam = Makeparam(gen)
        # predict
        return ( (makeparam.projector_real + makeparam.projector_reciprocal)*int(gen.getkw('npar')) + makeparam.wavefunction*float(gen.getkw('kpar')) )/1024.0/1024/1024 + int(gen.getkw('nnode'))*0.7
        # # warn
        # memory_available = int(gen.getkw('nnode')) * int(gen.getkw('mem_node'))
        # if memory_required > memory_available:
        #     print gen.__class__.__name__ + ' check_memory warning: insufficient memory. Mem required is {%s} GB. Available mem is {%s} GB.' %(memory_required, memory_available)
        # else:
        #     print gen.__class__.__name__ + ' check_memory report: Mem required is {%s} GB. Available mem is {%s} GB.' %(memory_required, memory_available)
