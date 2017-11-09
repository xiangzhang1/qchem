from scipy import spatial

dynamic.global_load()
m = dynamic.MLS['MLVASPSPEED']

_X = m.X_pipeline.transform(m._X)
_y0 = m.y_pipeline.transform(m._y0)

_X_dist = spatial.distance.pdist(_X)
_y0_dist = spatial.distance.pdist(_y0)

# problems identified. cannot guess why.
