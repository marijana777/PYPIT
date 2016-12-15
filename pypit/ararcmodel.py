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
    """ Add to existing lamp lists
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


def model_arcspec(lamps, wvcen, deltawv, npix, stretch, fwhm):
    """
    Parameters
    ----------
    lamps
    wvcen
    deltawv : float
      Total wavelength coverage in Ang, e.g. 1500A
    npix
    stretch
    fwhm

    Returns
    -------

    """
    # Load lines
    arc_tbl = load_lamps(lamps)

    # Generate wavelength array
    msgs.work("Deal with stretch")

    # Dispersion
    dwv = deltawv/npix
    wave = dwv*np.arange(npix) + (wvcen - dwv*npix/2)

    # Generate lines

