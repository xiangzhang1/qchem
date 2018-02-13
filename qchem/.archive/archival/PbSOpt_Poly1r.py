from scipy import spatial, optimize
from numpy.polynomial import polynomial as P
np.set_printoptions(precision=4, linewidth=250)
dynamic.global_load()

n = engine.Map().lookup('master.PbS QD.bare qd testing.crunchit.4 opt')
bcoor = n.cell.ccoor
ecoor = n.vasp.optimized_cell.ccoor
natom0 = n.cell.stoichiometry.values()[0]

# original
coef0 = np.array([
    [1, 0, 0, 0, 0, 0.1],
    [1, 0, 0, 0, 0, 0.1],
    [-1, 0, 0, 0, 0, 0.1]
])
# after run on 4opt
coef0 = np.array([[ -6.6974,  -9.6091,  -6.9815,  -2.4546,  -0.6249,  -1.9247],
       [ 30.2056,  -1.5994,  19.4142,   4.8667,  29.2056,  29.3056],
       [ 28.2056, -14.5317,  28.6521,  29.2056,  29.2056,  28.7521]])
coef_shape = coef0.shape

def opte(coef, coef_shape=coef_shape, bcoor=bcoor, ecoor=ecoor, natom0=natom0):
    from scipy import spatial, optimize     # ipython limits
    coef = coef.reshape(coef_shape)

    ccoor_shape = bcoor.shape

    def coore(ccoor, ccoor_shape=ccoor_shape, natom0=natom0, coef=coef):
        def paire(r, coef):
            from numpy.polynomial import polynomial as P
            return np.sum(P.polyval(1.0/r, coef))
        ccoor = ccoor.reshape(ccoor_shape)
        pdistPbPb = spatial.distance.pdist(ccoor[:natom0])
        pdistSS = spatial.distance.pdist(ccoor[natom0:])
        cdistPbS = spatial.distance.cdist(ccoor[:natom0], ccoor[natom0:])

        result = paire(pdistPbPb, coef[0]) + paire(pdistSS, coef[1]) + paire(cdistPbS, coef[2])
        return result

    def coorf(ccoor, ccoor_shape=ccoor_shape, natom0=natom0, coef=coef):
        def pairf(x, coef):
            from numpy.polynomial import polynomial as P
            r = np.linalg.norm(x, keepdims=True, axis=-1)
            f = P.polyval(1.0/r, P.polyder(coef)) * -r**-2
            return np.sum(f / r * x, axis=1)
        ccoor = ccoor.reshape(ccoor_shape)
        cPb   = ccoor[:natom0]
        cS    = ccoor[natom0:]
        xPbPb = np.array([np.delete(cPb,i,axis=0)-c for i,c in enumerate(cPb)])
        xSS   = np.array([np.delete(cS,i,axis=0)-c for i,c in enumerate(cS)])
        xPbS  = np.array([cS-c for i,c in enumerate(cPb)])
        xSPb  = np.array([cPb-c for i,c in enumerate(cS)])
        fPb   = pairf(xPbPb, coef[0]) + pairf(xPbS, coef[2])
        fS    = pairf(xSS, coef[1]) + pairf(xSPb, coef[2])
        result = np.concatenate((fPb, fS), axis=0)
        return result.reshape(-1)

    result = optimize.minimize(coore, x0=bcoor, jac=coorf, options={'maxiter':1000, 'disp':True})
    rcoor = result.x.reshape(ccoor_shape)
    dc = ecoor - rcoor
    error = np.linalg.norm(dc)

    print 'coef %s | error %s' %(coef.reshape(-1), error)

    return error

result = optimize.minimize(opte, x0=coef0, method='Powell')
