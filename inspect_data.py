from scipy import spatial

dynamic.global_load()
m = dynamic.MLS['MLPBSOPT']

_X = m.X_pipeline.transform(m._X)
_y0 = m.y_pipeline.transform(m._y0)

_X_dist = spatial.distance.pdist(_X)
_y0_dist = spatial.distance.pdist(_y0)

# problems identified. cannot guess why. 数值启发方法.
Xdm = spatial.distance.squareform(_X_dist)
ydm = spatial.distance.squareform(_y0_dist)

for ix, iy in np.ndindex(Xdm.shape[0], Xdm.shape[1]):
    if Xdm[ix, iy] < 0.1 and ydm[ix, iy] > 0.3:
        # 抓到你了!
        n1 = engine.Map().lookup(m._cur[ix])
        n2 = engine.Map().lookup(m._cur[iy])
        IPython.embed(banner1=u'抓到你了！')
