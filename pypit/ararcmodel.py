from __future__ import (print_function, absolute_import, division, unicode_literals)

import numpy as np
import os
from pypit import arpca
from pypit import arparse as settings
from pypit import armsgs
from pypit import arsave
from pypit import arutils
from pypit.arplot import get_dimen as get_dimen
from pypit import ararclines
from pypit import arqa
from matplotlib import pyplot as plt

from astropy.table import Table, vstack


try:
    from xastropy.xutils import xdebug as debugger
except ImportError:
    import pdb as debugger

# Logging
msgs = armsgs.get_logger()

# Path
import pypit
model_path = pypit.__path__[0]+'/data/arc_lines/Model/'


def add_to_lists(tbl):
    """ Add to existing lamp lists (or generate a new one)
    EXPERTS ONLY

    Parameters
    ----------
    tbl

    """
    # Check input table
    req_keys = ['ion', 'wave', 'NIST', 'Origin', 'amp']
    tbl_keys = tbl.keys()
    for key in req_keys:
        try:
            assert key in tbl_keys
        except AssertionError:
            msgs.error("Key {:s} not include in input table!".format(key))

    # Loop on ions
    ions = np.unique(tbl['ion'].data)
    for ion in ions:
        # Parse
        idx = tbl['ion'] == ion
        sub_tbl = tbl[idx]
        # New Table?
        tbl_file = model_path+'{:s}_model.ascii'.format(ion)
        if not os.path.isfile(tbl_file):
            sub_tbl.write(tbl_file, format='ascii.fixed_width')
            msgs.info("Generating new arc model table: {:s}".format(tbl_file))
        else:
            msgs.warn("Not ready to update table")
            debugger.set_trace()


def load_lamps(lamps):
    """ Load arc line data for input lamps
    Parameters
    ----------
    lamps : list

    Returns
    -------
    arc_tbl : Table

    """
    tbls = []
    for lamp in lamps:
        # Read
        tbl_file = model_path+'{:s}_model.ascii'.format(lamp)
        tbls.append(Table.read(tbl_file))
    # Stack
    arc_tbl = vstack(tbls, join_type='outer')
    # Return
    return arc_tbl


def match_quad_to_list(spec_lines, line_list, wv_guess, dwv_guess,
                  tol=2., dwv_uncertainty=0.2):
    """
    Parameters
    ----------
    spec_lines : ndarray
      pixel space
    line_list
    tol

    Returns
    -------
    possible_matchtes : list
      list of indices of matching quads

    """
    # Setup spec
    npix = spec_lines[-1]-spec_lines[0]
    spec_values = (spec_lines[1:-1]-spec_lines[0])/(
        spec_lines[-1]-spec_lines[0])
    ftol = tol/npix
    #
    possible_start = np.where((line_list > wv_guess[0]) & (line_list < wv_guess[1]))[0]
    possible_matches = []
    for start in possible_start:
        #print("Trying {:g}".format(line_list[start]))
        # Find possible ends
        possible_ends = np.where( (line_list > line_list[start] + npix*dwv_guess*(1-dwv_uncertainty)) &
                                     (line_list < line_list[start] + npix*dwv_guess*(1+dwv_uncertainty)))[0]
        # Loop on ends
        for end in possible_ends:
            values = (line_list[start+1:end]-line_list[start]) / (
                line_list[end]-line_list[start])
            # Test
            tst0 = np.abs(values-spec_values[0]) < ftol
            tst1 = np.abs(values-spec_values[1]) < ftol
            #if np.abs(line_list[start]-6097.8) < 0.2:
            #    debugger.set_trace()
            if np.any(tst0) & np.any(tst1):
                possible_matches.append([start, start+1+np.where(tst0)[0][0],
                                       start+1+np.where(tst1)[0][0], end])
    # Return
    return possible_matches

def model_arcspec(lamps, wvcen, deltawv, npix, stretch,
                  fwhm=4., npad=100, const_kms=False):
    """
    Parameters
    ----------
    lamps
    wvcen
    deltawv : float
      Total wavelength coverage in Ang, e.g. 1500A
    npix
    stretch
    fwhm : float, optional

    Returns
    -------

    """
    # Load lines
    arc_tbl = load_lamps(lamps)

    # Generate wavelength array
    msgs.work("Deal with stretch")

    # Dispersion
    dwv = deltawv/npix

    # Wave to pix function (normalized?)
    #def f_wave_pix(wave, dwv):

    # Generate wavelengths
    wave = dwv*np.arange(npix) + (wvcen - dwv*npix/2)

    # PAD

    wvmin, wvmax = np.min(wave), np.max(wave)

    # Cut
    model_lines = (arc_tbl['wave'] > wvmin) & (arc_tbl['wave'] < wvmax)
    model_waves = model_lines['wave'].data[model_lines]

    # Generate lines -- Should Cython the following

