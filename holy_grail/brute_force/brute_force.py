import numpy as np
import test_arcy
import pdb
from matplotlib import pyplot as plt
import time
import os, sys

this_file = os.path.realpath(__file__)
this_path = this_file[:this_file.rfind('/')]
sys.path.append(os.path.abspath(this_path+'/../../src'))
# PYPIT stuff...
last_updated = "2 May 2016"
version = '0.6'
import ardebug
debug = ardebug.init()
import armsgs
msgs = armsgs.get_logger((None, debug, last_updated, version, 2))
import ararclines


def robust_polyfit(xarray, yarray, order, weights=None, maxone=True, sigma=3.0, function="polynomial", initialmask=None, forceimask=False, minv=None, maxv=None, debug=False, **kwargs):
    """
    A robust (equally weighted) polynomial fit is performed to the xarray, yarray pairs
    mask[i] = 1 are masked values

    :param xarray: independent variable values
    :param yarray: dependent variable values
    :param order: the order of the polynomial to be used in the fitting
    :param weights: weights to be used in the fitting (weights = 1/sigma)
    :param maxone: If True, only the most deviant point in a given iteration will be removed
    :param sigma: confidence interval for rejection
    :param function: which function should be used in the fitting (valid inputs: 'polynomial', 'legendre', 'chebyshev', 'bspline')
    :param initialmask: a mask can be supplied as input, these values will be masked for the first iteration. 1 = value masked
    :param forceimask: if True, the initialmask will be forced for all iterations
    :param minv: minimum value in the array (or the left limit for a legendre/chebyshev polynomial)
    :param maxv: maximum value in the array (or the right limit for a legendre/chebyshev polynomial)
    :param debug:
    :return: mask, ct -- mask is an array of the masked values, ct is the coefficients of the robust polyfit.
    """
    # Setup the initial mask
    if initialmask is None:
        mask = np.zeros(xarray.size, dtype=np.int)
        if forceimask:
            msgs.warn("Initial mask cannot be enforced -- no initital mask supplied")
            forceimask = False
    else:
        mask = initialmask.copy()
    mskcnt = np.sum(mask)
    # Iterate, and mask out new values on each iteration
    while True:
        w = np.where(mask == 0)
        xfit = xarray[w]
        yfit = yarray[w]
        if weights is not None:
            wfit = weights[w]
        else:
            wfit = None
        ct = func_fit(xfit, yfit, function, order, w=wfit, minv=minv, maxv=maxv, **kwargs)
        yrng = func_val(ct, xarray, function, minv=minv, maxv=maxv)
        sigmed = 1.4826*np.median(np.abs(yfit-yrng[w]))
        if debug:
            import pdb
            pdb.set_trace()
        if xarray.size-np.sum(mask) <= order+2:
            msgs.warn("More parameters than data points - fit might be undesirable")
            break  # More data was masked than allowed by order
        if maxone:  # Only remove the most deviant point
            tst = np.abs(yarray[w]-yrng[w])
            m = np.argmax(tst)
            if tst[m] > sigma*sigmed:
                mask[w[0][m]] = 1
        else:
            if forceimask:
                w = np.where((np.abs(yarray-yrng) > sigma*sigmed) | (initialmask==1))
            else:
                w = np.where(np.abs(yarray-yrng) > sigma*sigmed)
            mask[w] = 1
        if mskcnt == np.sum(mask): break  # No new values have been included in the mask
        mskcnt = np.sum(mask)
        w = np.where(mask == 0)
    # Final fit
    xfit = xarray[w]
    yfit = yarray[w]
    if weights is not None:
        wfit = weights[w]
    else:
        wfit = None
    ct = func_fit(xfit, yfit, function, order, w=wfit, minv=minv, maxv=maxv, **kwargs)
    return mask, ct

def func_fit(x, y, func, deg, minv=None, maxv=None, w=None, **kwargs):
    if func == "polynomial":
        return np.polynomial.polynomial.polyfit(x, y, deg, w=w)
    elif func == "legendre":
        if minv is None or maxv is None:
            if np.size(x) == 1:
                xmin, xmax = -1.0, 1.0
            else:
                xmin, xmax = np.min(x), np.max(x)
        else:
            xmin, xmax = minv, maxv
        xv = 2.0 * (x-xmin)/(xmax-xmin) - 1.0
        return np.polynomial.legendre.legfit(xv, y, deg, w=w)
    elif func == "chebyshev":
        if minv is None or maxv is None:
            if np.size(x) == 1:
                xmin, xmax = -1.0, 1.0
            else:
                xmin, xmax = np.min(x), np.max(x)
        else:
            xmin, xmax = minv, maxv
        xv = 2.0 * (x-xmin)/(xmax-xmin) - 1.0
        return np.polynomial.chebyshev.chebfit(xv, y, deg, w=w)
    elif func == "bspline":
        return bspline_fit(x, y, order=deg, w=w, **kwargs)
    else:
        msgs.error("Fitting function '{0:s}' is not implemented yet" + msgs.newline() +
                   "Please choose from 'polynomial', 'legendre', 'chebyshev','bspline'")


def func_val(c, x, func, minv=None, maxv=None):
    if func == "polynomial":
        return np.polynomial.polynomial.polyval(x, c)
    elif func == "legendre":
        if minv is None or maxv is None:
            if np.size(x) == 1:
                xmin, xmax = -1.0, 1.0
            else:
                xmin, xmax = np.min(x), np.max(x)
        else:
            xmin, xmax = minv, maxv
        xv = 2.0 * (x-xmin)/(xmax-xmin) - 1.0
        return np.polynomial.legendre.legval(xv, c)
    elif func == "chebyshev":
        if minv is None or maxv is None:
            if np.size(x) == 1:
                xmin, xmax = -1.0, 1.0
            else:
                xmin, xmax = np.min(x), np.max(x)
        else:
            xmin, xmax = minv, maxv
        xv = 2.0 * (x-xmin)/(xmax-xmin) - 1.0
        return np.polynomial.chebyshev.chebval(xv, c)
    elif func == "bspline":
        return interpolate.splev(x, c, ext=1)
    else:
        msgs.error("Fitting function '{0:s}' is not implemented yet" + msgs.newline() +
                   "Please choose from 'polynomial', 'legendre', 'chebyshev', 'bspline'")


#############
# Define the problem

npix = 2048.0

pixels = np.array([  140.93689967,   154.42571123,   194.07916869,   205.86298773,
         212.86313041,   253.05224277,   272.55370351,   307.26868399,
         335.39219804,   349.2515197 ,   378.94265744,   391.93324683,
         425.84238777,   456.89126564,   481.01486112,   499.6324469 ,
         530.18107916,   542.27304318,   607.60782398,   624.11784818,
         665.38725867,   714.8788845 ,   739.04367662,   871.12853433,
         893.42568193,   934.96578041,   956.46186979,  1005.91212302,
        1022.5280144 ,  1066.56388483,  1083.79186726,  1152.3883973 ,
        1186.31546381,  1217.15943528,  1233.12444534,  1246.16520617,
        1278.06325613,  1286.78211123,  1344.23944597,  1443.37024723,
        1501.14774051,  1537.09369339,  1651.17129411,  1677.04690861,
        1718.40233216,  1767.86758504,  1787.00940503,  1849.84686133,
        1857.67905523,  2018.04455387])

waves = np.array([    0.    ,     0.    ,     0.    ,     0.    ,     0.    ,
           0.    ,     0.    ,     0.    ,     0.    ,     0.    ,
           0.    ,     0.    ,     0.    ,     0.    ,     0.    ,
           0.    ,     0.    ,     0.    ,     0.    ,     0.    ,
           0.    ,     0.    ,     0.    ,     0.    ,     0.    ,
           0.    ,     0.    ,  7149.012 ,  7175.9154,  7247.1631,
        7274.94  ,  7386.014 ,  7440.9469,  7490.9335,  7516.721 ,
        7537.8488,  7589.5025,  7603.6384,     0.    ,     0.    ,
           0.    ,     0.    ,     0.    ,     0.    ,     0.    ,
           0.    ,     0.    ,     0.    ,     0.    ,     0.    ])

mask = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])

# Load the linelist
ions = ['HgI','ArI','NeI','XeI','KrI']
linelist = ararclines.load_arcline_list(None, 0, ions, '600/7500', wvmnx=None)
ll = linelist['wave'].data
linelist = np.array(ll[np.where(~ll.mask)])
linelist.sort()

# Fit the known pixels with a cubic polynomial
yval = pixels/npix
wmsk = np.where(mask==0)

# Determine (roughly) the minimum and maximum wavelengths
coeff = func_fit(yval[wmsk], waves[wmsk], "polynomial", 1)
wmnmx = func_val(coeff, np.array([0.0,1.0]), "polynomial")
wmin, wmax = wmnmx[0], wmnmx[1]

wavmean = np.mean(waves[wmsk])
xval = (waves - wavmean)/(wmax-wmin)
ll = (linelist - wavmean)/(wmax-wmin)
# Find the first and second coefficients of the polynomial fitting
# These coefficients are given by (p0) the value of the cubic fit
# at the mean wavelength of the id'ed pixels, and (p1) the value of
# the derivative of the cubic fit at the mean wavelength of the id'ed
# pixels.
coeff = func_fit(xval[wmsk], yval[wmsk], "polynomial", 3)
# ... and we only need the first two coefficients
coeff = coeff[:2]

# Now send the data away to cython to determine the best solution
# for the remaining two cofficients by brute force
print "Commencing brute Force identification"
start= time.time()
lim=0.3
wdiff = (wmax-wmin)*lim
wll = np.where((linelist > wmin-wdiff) & (linelist < wmax+wdiff))[0]
wavidx = test_arcy.brute_force_solve(yval, ll[wll], coeff, npix, lim)
end = time.time()
print "Execution time:", end-start
#print wavidx

wavids = linelist[wll[wavidx]]

# Fit the best solution
xmod = np.arange(npix)
coeff = func_fit(pixels, wavids, "polynomial", 3)
coeffb = func_fit(wavids, pixels, "polynomial", 3)
ymod = func_val(coeff, xmod, "polynomial")
pmod = func_val(coeff, pixels, "polynomial")
wmod = func_val(coeffb, wavids, "polynomial")
plt.subplot(211)
plt.plot(pixels, wavids, 'bx')
plt.plot(xmod, ymod, 'r-')
plt.subplot(212)
#plt.plot(pixels, wavids-pmod, 'bx')
plt.plot(pixels-wmod, wavids, 'bx')
#plt.plot(np.array([0.0,npix]), np.zeros(2), 'r-')
plt.plot(np.zeros(2), np.array([3300.0,10000.0]), 'r-')
plt.show()
print "Coefficients:", coeff
#print wavids
print "Pixel residual:", np.std(pixels-wmod)