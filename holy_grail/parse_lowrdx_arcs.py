""" Generate hdf5 files from LowRedux save files
"""
from __future__ import print_function, absolute_import, division, unicode_literals
import linetools.utils as ltu

import sys, os, json
import numpy as np
import h5py
from scipy.io.idl import readsav

from matplotlib import pyplot as plt

from astropy.table import Table
from astropy import units as u

this_file = os.path.realpath(__file__)
this_path = this_file[:this_file.rfind('/')]
sys.path.append(os.path.abspath(this_path+'/../src'))
sys.path.append(os.path.abspath(this_path+'/../holy_grail'))
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
import arutils
import arholy
import arwave

import test_holy

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


def generate_hdf(sav_file, lamps, outfil, dtoler=0.6):
    """ Given an input LR IDL save file, generate an hdf5
    IDs arc lines too

    Parameters
    ----------
    sav_file : str
      Root name of the IDL save file from LowRedux, e.g. lris_blue_600.sav
    lamps
    outfil

    Returns
    -------

    """
    # Read IDL save file
    sav_file = os.getenv('LONGSLIT_DIR')+'calib/linelists/'+sav_file
    s = readsav(sav_file)
    ctbl = Table(s['calib'])  # For writing later

    # Line list
    #air_alist = ararclines.load_arcline_list(None,None, lamps, None, use_vac=False)
    alist = ararclines.load_arcline_list(None,None, lamps, None)

    # Meta data
    odict = dict(npix=len(s['archive_arc'][0]),
                 lamps=[str(ilamp) for ilamp in lamps],  # For writing to hdf5
                 nspec=len(s['archive_arc']), infil=sav_file, IDairvac='vac')

    # Start output
    outh5 = h5py.File(outfil, 'w')
    # Meta data
    outh5.create_group('meta')
    for key in odict.keys():
        try:
            outh5['meta'][key] = odict[key]
        except TypeError:  # Probably a unicode thing
            debugger.set_trace()
    outh5.create_group('arcs')

    # Loop on spectra
    for ss in range(odict['nspec']):
        sss = str(ss)
        # Spec
        spec = s['archive_arc'][ss]
        # Peaks
        tampl, tcent, twid, w, yprep = find_peaks(spec)
        pixpk = tcent[w]
        pixampl = tampl[w]

        # Wavelength solution
        wv_air = cheby_val(s['calib'][ss]['ffit'], np.arange(odict['npix']),
                   s['calib'][ss]['nrm'],s['calib'][ss]['nord'])
        # Peak waves
        twave_air = cheby_val(s['calib'][ss]['ffit'], pixpk,
                              s['calib'][ss]['nrm'],s['calib'][ss]['nord'])
        # Air to Vac
        twave_vac = arwave.airtovac(twave_air*u.AA)
        # IDs
        idwv = np.zeros_like(pixpk)
        for kk,twv in enumerate(twave_vac.value):
            # diff
            diff = np.abs(twv-alist['wave'])
            if np.min(diff) < dtoler:
                idwv[kk] = alist['wave'][np.argmin(diff)]
        # Output
        outh5['arcs'].create_group(sss)
        # Datasets
        outh5['arcs'][sss]['spec'] = spec
        outh5['arcs'][sss]['pixpk'] = pixpk
        outh5['arcs'][sss]['ID'] = idwv
        outh5['arcs'][sss]['ID'].attrs['airvac'] = 'vac'
        # LR wavelengths
        outh5['arcs'][sss]['LR_wave'] = wv_air
        outh5['arcs'][sss]['LR_wave'].attrs['airvac'] = 'air'
        # LR Fit
        outh5['arcs'][sss].create_group('LR_fit')
        for key in ctbl.keys():
            outh5['arcs'][sss]['LR_fit'][key] = ctbl[ss][key]
    # Close
    outh5.close()
    print('Wrote {:s}'.format(outfil))

# Command line execution
if __name__ == '__main__':

    # LRISb 600
    generate_hdf('lris_blue_600.sav', ['ZnI', 'CdI', 'HgI'], 'test_arcs/LRISb_600.hdf5')

