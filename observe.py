#data
_X = np.c_[np.arange(3, 20, 0.1)]
_y0 = np.c_[1.0 / _X ** 2]

# pipeline
_X_pipeline = StandardScaler()
_y0_pipeline = StandardScaler()

_X_scaled = _X_pipeline.fit_transform(_X)
_y0_scaled = _y0_pipeline.fit_transform(_y0)

# ann
net = Sequential(
nn.Linear(1,5),
nn.ELU(),
nn.Linear(5,5),
nn.ELU(),
nn.Linear(5,1)
)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# batch
ds = TensorDataset(torch.FloatTensor(_X_scaled), torch.FloatTensor(_y0_scaled))
dl = DataLoader(ds, batch_size=32, shuffle=True)

# train
net.train()
n_epochs = 1000
for epoch in tqdm(range(n_epochs)):
    for _X_batch, _y0_batch in dl:
        X, y0 = Variable(_X_batch, requires_grad=True), Variable(_y0_batch, requires_grad=False)
        y = net(X)
        net.zero_grad()
        loss = criterion(y, y0)
        loss.backward()
        optimizer.step()

# test
net.eval()
X = Variable(torch.FloatTensor(_X_scaled), requires_grad=True)
y = net(X)
_y_scaled = y.data.numpy()
_y = _y0_pipeline.inverse_transform(_y_scaled)

plt.plot(_X, _y, '-', _X, _y0, '--')
plt.show()
