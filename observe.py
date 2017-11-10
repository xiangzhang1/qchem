_X = np.c_[np.arange(3, 20, 0.1)]
_y0 = np.c_[1.0 / _X ** 2]

_X_scaler = StandardScaler()
_y0_scaler = StandardScaler()

_X_scaled = _X_scaler.fit_transform(_X)
_y0_scaled = _y0_scaler.fit_transform(_y0)

net = Sequential(
nn.Linear(1,5),
nn.ReLU(),
nn.Linear(5,5),
nn.ReLU(),
nn.Linear(5,1)
)
net.train()
criterion = nn.MSELoss()
optimizer = optim.SGD()

ds = TensorDataset(_X_scaled, _y0_scaled)
dl = DataLoader(ds, batch_size=32)

n_epochs = 1000
for epoch in range(n_epochs):
    for _X_batch, _y0_batch in dl:
        X, y0 = Variable(_X_batch, requires_grad=True), Variable(_y0_batch, requires_grad=True)
        y = net(X)
        loss = criterion(y, y0)
        loss.backward()
        net.zero_grad()
        optimizer.step()
