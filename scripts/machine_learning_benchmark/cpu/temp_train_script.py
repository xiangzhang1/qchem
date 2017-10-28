data = []
with open("/home/xzhang1/m_cpu_config.log", "r") as if_:
    lines = if_.readlines()
    for line in lines:
        if line and not line.startswith('#') and len(line.split())==len(lines[1].split()):
            data.append( np.float_(line.split()) )
data = np.float_(data)
data[:, -2] *= 10**6

shared.ML_VASP_MEMORY = engine.Ml_vasp_memory()
m = shared.ML_VASP_MEMORY
X = data[:, :-3]
Y = data[:, -2]
m.X_train = X
m.Y_train = Y
m.scale_and_fit()
Y_pred = m.scale_and_predict(X)
np.savetxt(sys.stdout, zip(Y, Y_pred), fmt='%s', delimiter=' ')
