from scipy import spatial

dynamic.global_load()
m = dynamic.MLS['MLPBSOPT']

_X_local = np.float32(m._X_local)
_X_high = m.X_high_pipeline.fit_transform(m._X_high)
_X_global = m.X_high_pipeline.fit_transform(np.float32(m._X_global).reshape((_X_local.shape[0], -1)))
_y0 = m.y_pipeline.transform(m._y0)

Xdist = spatial.distance.pdist(_X_global)    #np.concatenate((_X_local, _X_high), axis=1)
ydist = spatial.distance.pdist(_y0)

# problems identified. cannot guess why. 数值启发方法.
Xdm = spatial.distance.squareform(np.concatenate((_X_local, _X_high), axis=1))
ydm = spatial.distance.squareform(_y0_dist)

for ix, iy in np.ndindex(Xdm.shape[0], Xdm.shape[1]):
    if Xdm[ix, iy] < 0.1 and ydm[ix, iy] > 0.3:
        # 抓到你了!
        n1 = engine.Map().lookup(m._cur[ix])
        n2 = engine.Map().lookup(m._cur[iy])
        IPython.embed(banner1=u'抓到你了！')
