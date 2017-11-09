from scipy import spatial

dynamic.global_load()
m = MLS['MLVASPSPEED']

_X = m.X_pipeline.transform(m._X)
_y0 = m.y0_pipeline.transform(m._y0)

_X_dist = sptial.distance.pdist(_X)
_y0_dist = sptial.distance.pdist(_y0)
