# Tests for the Holy Grail
from __future__ import print_function, absolute_import, division, unicode_literals
import linetools.utils as ltu

import sys, os, json
import h5py
import numpy as np
from scipy.io.idl import readsav

import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

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
import arutils
import arholy
import arwave
import test_holy

try:
    from xastropy.xutils import xdebug as debugger
except:
    import pdb as debugger

def load_test_file(tst_file):
    """ Read hdf5 test file
    Returns
    -------

    """
    hdf5 = h5py.File(tst_file, 'r')
    # Grab meta data
    mdict = {}
    for key in hdf5['meta'].keys():
        mdict[key] = hdf5['meta'][key].value
    mdict['lamps'] = list(mdict['lamps'])
    # Spectra (as a list)
    arcs = []
    for ii in range(mdict['nspec']):
        sii = str(ii)
        tdict = {}
        for key in hdf5['arcs'][sii].keys():
            # Groups
            if key in ['LR_fit']:
                tdict[key] = {}
                for ikey in hdf5['arcs'][sii][key].keys():
                    tdict[key][ikey] = hdf5['arcs'][sii][key][ikey].value
            else:
                tdict[key] = hdf5['arcs'][sii][key].value
        # Append
        arcs.append(tdict.copy())
    hdf5.close()
    # Return
    return mdict, arcs

def main(tst_file, ntrial, seed=1234, use_holy1=False):
    """
    Parameters
    ----------
    tst_file
    ntrial
    seed

    Returns
    -------

    """
    rstate = np.random.RandomState(seed)
    # Load test
    meta, arcs = load_test_file(tst_file)
    # Load linelist
    alist = ararclines.load_arcline_list(None,None, meta['lamps'], None)

    # Setup Random bits and pieces
    # Random arcs
    ranarc = rstate.randint(0, len(arcs), size=ntrial)

    # Holy 1
    if not use_holy1:
        ran_nline = rstate.randint(3, 6, size=ntrial)
        ran_icen = meta['npix']*(0.1 + 0.8*rstate.rand(ntrial))

    # Meta plot
    # Run
    for ss in range(ntrial):
        # Grab random arc
        arc = arcs[ranarc[ss]]
        # Init
        ids = arc['ID'] > 0.
        all_idpix = arc['pixpk'][ids]
        all_idwv = arc['ID'][ids]
        # HOLY 1
        if not use_holy1:
            icen = ran_icen[ss]
            nline = ran_nline[ss]
            idpix, idwave = test_holy.grab_id_lines(all_idpix, all_idwv, icen, nline)
            debugger.set_trace()


# Command line execution
if __name__ == '__main__':
    tst_file = 'test_arcs/LRISb_600.hdf5'
    main(tst_file, 10)