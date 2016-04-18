# Tests for the Holy Grail
from __future__ import print_function, absolute_import, division, unicode_literals

import sys, os
import numpy as np
from scipy.io.idl import readsav

from matplotlib import pyplot as plt

this_file = os.path.realpath(__file__)
this_path = this_file[:this_file.rfind('/')]
sys.path.append(os.path.abspath(this_path+'/../src'))
import armsgs as msgs

import ardebug
debug = ardebug.init()
last_updated = "26 November 2015"
version = '0.3'
verbose = False
msgs = msgs.get_logger((None, debug, last_updated, version, verbose))

import ararc
import ararclines
import arcyarc

try:
    from xastropy.xutils import xdebug as debugger
except:
    import pdb as debugger

def fcheby(xnrm,order):
    leg = np.zeros((len(xnrm),order))
    leg[:,0] = 1.
    if order >= 2:
        leg[:,1] = xnrm
    # For looop
    for j in range(2,order):
        leg[:,j] = 2.0 * xnrm * leg[:,j-1] - leg[:,j-2]
    # Return
    return leg

def cheby_val(coeff, x, nrm, order):
    #
    xnrm = 2. * (x - nrm[0])/nrm[1]
    # Matrix first
    leg = fcheby(xnrm, order)
    # Dot
    return np.dot(leg, coeff)

#################
def find_peaks(censpec):
    siglev = 6.0
    bpfit = 5  # order of the polynomial used to fit the background 'continuum'
    fitp = 7 #slf._argflag['arc']['calibrate']['nfitpix']
    if len(censpec.shape) == 3: detns = censpec[:, 0].flatten()
    else: detns = censpec.copy()
    xrng = np.arange(float(detns.size))
    yrng = np.zeros(detns.size)
    mask = np.zeros(detns.size, dtype=np.int)
    mskcnt = 0
    while True:
        w = np.where(mask == 0)
        xfit = xrng[w]
        yfit = detns[w]
        ct = np.polyfit(xfit, yfit, bpfit)
        yrng = np.polyval(ct, xrng)
        sigmed = 1.4826*np.median(np.abs(detns[w]-yrng[w]))
        w = np.where(detns > yrng+1.5*sigmed)
        mask[w] = 1
        if mskcnt == np.sum(mask):
            break  # No new values have been included in the mask
        mskcnt = np.sum(mask)
    #
    w = np.where(mask == 0)
    xfit = xrng[w]
    yprep = detns - yrng
    sfit = 1.4826*np.abs(detns[w]-yrng[w])
    ct = np.polyfit(xfit, sfit, bpfit)
    yerr = np.polyval(ct, xrng)
    myerr = np.median(np.sort(yerr)[:yerr.size/2])
    yerr[np.where(yerr < myerr)] = myerr
    # Find all significant detections
    # The last argument is the overall minimum significance level of an arc line detection and the second
    # last argument is the level required by an individual pixel before the neighbourhood of this pixel is searched.
    satsnd = np.zeros_like(censpec)
    tpixt, num = arcyarc.detections_sigma(yprep, yerr, np.zeros(satsnd.shape[0], dtype=np.int), siglev/2.0, siglev)
    pixt = arcyarc.remove_similar(tpixt, num)
    pixt = pixt[np.where(pixt != -1)].astype(np.int)
    tampl, tcent, twid, ngood = arcyarc.fit_arcorder(xrng, yprep, pixt, fitp)
    w = np.where((np.isnan(twid) == False) & (twid > 0.0) & (twid < 10.0/2.35) & (tcent > 0.0) & (tcent < xrng[-1]))
    # Return
    return tampl, tcent, twid, w, yprep


def test_lrisr_600_7500(debug=True):
    """ Tests on the IDL save file for LRISr
    Returns
    -------

    """
    id_wave = [6506.528 ,  6678.2766,  6717.043 ,  6929.4672,  6965.431]
    llist = ararclines.load_arcline_list(None,None,
                                         ['ArI','NeI','HgI','KrI','XeI'],None)

    # IDL save file
    sav_file = os.getenv('LONGSLIT_DIR')+'calib/linelists/lris_red_600_7500.sav'
    s = readsav(sav_file)

    idx = 0
    spec = s['archive_arc'][idx]
    npix = len(spec)

    # Find peaks
    tampl, tcent, twid, w, yprep = find_peaks(spec)
    pixpk = tcent[w]
    pixampl = tampl[w]
    # Saturation here
    if False:
        plt.clf()
        ax = plt.gca()
        ax.plot(np.arange(npix), yprep, 'k', drawstyle='mid-steps')
        ax.scatter(pixpk, pixampl, marker='o')
        plt.show()
        debugger.set_trace()

    # Evaluate fit at peaks
    pixwave = cheby_val(s['calib'][idx]['ffit'], pixpk,
                        s['calib'][idx]['nrm'],s['calib'][idx]['nord'],)
    # Setup IDlines
    id_pix = []
    for idw in id_wave:
        diff = np.abs(idw-pixwave)
        imin = np.argmin(diff)
        if diff[imin] < 2.:
            id_pix.append(pixpk[imin])
        else:
            raise ValueError("No match to {:g}!".format(idw))
    idlines = ararc.IDLines(np.array(id_pix), np.array(id_wave))

    # Holy Grail
    ararc.searching_for_the_grail(pixpk=pixpk,idlines=idlines, npix=npix, llist=llist,
                                  extrap_off=750.)
    # PDF
    debugger.set_trace()



#################
def main(flg_test):

    if flg_test == 'all':
        flg_test = np.sum( np.array( [2**ii for ii in range(15)] ))
    else:
        flg_test = int(flg_test)

    # Fiducial
    if (flg_test % 2**1) >= 2**0:
        test_lrisr_600_7500()

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_test = 0
        flg_test += 2**0   # LRISr
    else:
        flg_fig = sys.argv[1]

    main(flg_test)
