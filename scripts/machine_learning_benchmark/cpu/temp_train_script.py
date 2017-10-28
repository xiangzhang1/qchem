data = []
with open("m_cpu_config.log", "r") as if_:
    lines = if_.readlines()
    for line in lines:
        if line and not line.startswith('#') and len(line.split())==len(lines[1].split()):
            data.append( np.float_(line.split()) )
data = np.float_(data)

shared.ML_VASP_MEMORY.X = data[:, :-3]
shared.ML_VASP_MEMORY.Y = data[:, -2]
