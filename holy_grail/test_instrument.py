# Tests for the Holy Grail
from __future__ import print_function, absolute_import, division, unicode_literals


import sys, os, time
import h5py
import numpy as np

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


def load_arc_file(arc_file):
    """ Read hdf5 arc file
    Returns
    -------

    """
    if arc_file.find('/') < 0:
        path = this_path+'/test_arcs/'
    hdf5 = h5py.File(path+arc_file, 'r')
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


def output_test_file(meta, test_dicts, outfil):
    """
    Parameters
    ----------
    meta
    trial_dicts
    outfil : str

    Returns
    -------

    """
    # Start output
    outh5 = h5py.File(outfil, 'w')
    # Meta data
    outh5.create_group('meta')
    for key in meta.keys():
        try:
            outh5['meta'][key] = meta[key]
        except TypeError:  # Probably a unicode thing
            debugger.set_trace()
    # Trials
    outh5.create_group('trials')
    for ii in range(meta['ntrials']):
        sii = str(ii)
        outh5['trials'].create_group(sii)
        # Dump the dict
        tdict = test_dicts[ii]
        for key in tdict.keys():
            outh5['trials'][sii][key] = tdict[key]
    # Finish
    outh5.close()
    print("Writing {:s}".format(outfil))


def load_test_file(test_file):
    """ Read hdf5 test file
    Returns
    -------

    """
    hdf5 = h5py.File(test_file, 'r')
    # Grab meta data
    mdict = {}
    for key in hdf5['meta'].keys():
        mdict[key] = hdf5['meta'][key].value
    # Spectra (as a list)
    test_dicts = []
    for ii in range(mdict['ntrials']):
        sii = str(ii)
        tdict = {}
        for key in hdf5['trials'][sii].keys():
            # Groups
            '''
            if key in ['LR_fit']:
                tdict[key] = {}
                for ikey in hdf5['arcs'][sii][key].keys():
                    tdict[key][ikey] = hdf5['arcs'][sii][key][ikey].value
            else:
            '''
            tdict[key] = hdf5['trials'][sii][key].value
        # Append
        test_dicts.append(tdict.copy())
    hdf5.close()
    # Return
    return mdict, test_dicts


def plot_meta(pp, meta, arcs):
    """ Plot an arc spectrum and meta data
    Parameters
    ----------
    ax
    meta
    spec

    Returns
    -------

    """
    # Init
    idxa = 0
    arc = arcs[idxa]

    # Initialize
    #xmnx = (0.1, 1.5)
    #ymnx = (-0.01, 0.01)

    # Start the plot
    fig = plt.figure(figsize=(8, 5))
    plt.clf()
    ax = plt.gca()

    lsz = 12.

    # Spectrum
    ax.plot(arc['wave'], arc['spec'], 'b', drawstyle='mid-steps')
    ax.set_xlabel('Wavelength (Ang)')
    ax.set_ylabel('Counts')

    # IDs
    ids = arc['ID'] > 0.
    all_idpix = arc['pixpk'][ids]
    all_idwv = arc['ID'][ids]
    all_ions = arc['Ion'][ids]
    ymin, ymax = 0., np.max(arc['spec'])
    ysep = ymax*0.03
    for kk, pix in enumerate(all_idpix):
        yline = np.max(arc['spec'][int(pix)-2:int(pix)+2])
        x = all_idwv[kk]
        # Tick mark
        ax.plot([x,x], [yline+ysep*0.25, yline+ysep], 'g-')
        # label
        ax.text(x, yline+ysep*1.3,
                '{:s} {:g}'.format(all_ions[kk], all_idwv[kk]),
                ha='center', va='bottom', size='small', rotation=90., color='green')

    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig(bbox_inches='tight')
    plt.close()

def plot_holy2(pp, arcs, test_dicts):
    """
    Parameters
    ----------
    pp
    arcs
    tests

    Returns
    -------

    """
    # Init
    ntrials = len(test_dicts)
    rand_off = np.random.rand(ntrials)
    NIDs = np.zeros(ntrials)
    NBAD = np.zeros(ntrials)
    NGOOD = np.zeros(ntrials)
    wavec = np.zeros(ntrials)
    NHOLY1 = np.zeros(ntrials)
    for ii in range(ntrials):
        NIDs[ii] = test_dicts[ii]['NID']
        NBAD[ii] = test_dicts[ii]['NBAD']
        NGOOD[ii] = test_dicts[ii]['NGOOD']
        wavec[ii] = np.mean(test_dicts[ii]['idwv']) + 5.*rand_off[ii]
        NHOLY1[ii] = len(test_dicts[ii]['idpix_1']) + 0.5*(rand_off[ii]-0.5)
    #
    plt.figure(figsize=(6, 5))
    plt.clf()
    gs = gridspec.GridSpec(2, 2)

    # NID
    ax = plt.subplot(gs[0, 0])
    xmnx = [np.min(NIDs), np.max(NIDs)]
    binsz=2
    if (xmnx[1]-xmnx[0]) % 2 == 1:
        xmnx[1] += 1
    nbin = (xmnx[1]-xmnx[0])/binsz
    hist, edges = np.histogram(NIDs, range=xmnx, bins=nbin)
    ax.bar(edges[:-1], hist, width=binsz)
    ax.set_xlabel('NID')
    ax.set_ylabel('Number')

    # NBAD
    ax = plt.subplot(gs[0, 1])
    xmnx = np.min(NBAD), np.max(NBAD)
    binsz=1
    nbin = xmnx[1]+1
    hist, edges = np.histogram(NBAD, range=(xmnx[0]-0.5, xmnx[1]+0.5),
                                   bins=nbin)
    ax.bar(edges[:-1], hist, width=binsz, color='red')
    ax.set_xlabel('NBAD')
    ax.set_ylabel('Number')

    #NGOOD vs. wvc
    xtick = np.round((np.max(wavec)-np.min(wavec))/3,-2)
    ax = plt.subplot(gs[1, 0])
    ax.scatter(wavec, NGOOD)
    ax.set_xlabel(r'$\lambda_c$')
    ax.set_ylabel('NGOOD')
    ax.xaxis.set_major_locator(plt.MultipleLocator(xtick))

    #NGOOD vs. NHOLY1
    ax = plt.subplot(gs[1, 1])
    ax.scatter(NHOLY1, NGOOD)
    ax.set_xlabel('NHOLY1')
    ax.set_ylabel('NGOOD')

    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig(bbox_inches='tight')
    plt.close()


def fig_holytest(arc_file, test_file, outfil):
    """
    Parameters
    ----------
    arc_file
    test_file
    outfil

    Returns
    -------

    """
    ameta, arcs = load_arc_file(arc_file)
    tmeta, test_dicts = load_test_file(test_file)

    # Start plot
    pp = PdfPages(outfil)

    # Meta
    plot_meta(pp, ameta, arcs)
    # HOLY 2
    plot_holy2(pp, arcs, test_dicts)

    # Finish
    pp.close()
    print("Writing {:s}".format(outfil))


def set_ids_holy2(tcent, all_idpix, all_idwv):
    # Mask ID
    tmsk = tcent==tcent
    twv = np.zeros_like(tcent)
    for ii,itc in enumerate(tcent):
        if np.min(np.abs(itc-all_idpix)) > 0.1:
            tmsk[ii] = False
        else:
            twv[ii] = all_idwv[np.argmin(np.abs(itc-all_idpix))]
    # Return
    return tmsk, twv


def main(arc_file, ntrials, outfil, seed=1234, use_holy1=False, verbose=False,
         ngrid=500):
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
    ameta, arcs = load_arc_file(arc_file)
    # Load linelist
    alist = ararclines.load_arcline_list(None,None, ameta['lamps'], None)
    llist = np.array(alist['wave'])
    llist.sort()

    # Meta dict
    mdict = {}
    mdict['arc_file'] = arc_file
    mdict['ntrials'] = ntrials
    mdict['time'] = time.strftime("%d%b%Y_%Hh%Mm%Ss", time.localtime())
    mdict['lamps'] = ameta['lamps']
    mdict['instr'] = ameta['instr']

    # Setup Random bits and pieces
    # Random arcs
    ranarc = rstate.randint(0, len(arcs), size=ntrials)

    # Holy 1
    if not use_holy1:
        ran_nline = rstate.randint(3, 6, size=ntrials)
        ran_icen = ameta['npix']*(0.1 + 0.8*rstate.rand(ntrials))

    # Meta plot
    # Run
    test_dicts = []
    for ss in range(ntrials):
        # Grab random arc
        arc = arcs[ranarc[ss]]
        # Init
        tdict = dict(idx=ss, arcid=ranarc[ss])
        ids = arc['ID'] > 0.
        all_idpix = arc['pixpk'][ids]
        all_idwv = arc['ID'][ids]
        tdict['all_idpix'] = all_idpix
        tdict['all_idwv'] = all_idwv
        # HOLY 1
        if not use_holy1:
            tcent = arc['pixpk']
            icen = ran_icen[ss]
            nline = ran_nline[ss]
            idpix, idwave = test_holy.grab_id_lines(all_idpix, all_idwv, icen, nline)
            tdict['icen'] = icen
            tdict['nline'] = nline
        tdict['idpix_1'] = idpix
        tdict['idwv_1'] = idwave
        if nline < 4:  # HOLY 1b
            idpix,idwave = arholy.add_lines(tcent, idpix, idwave, llist, verbose=verbose)
            tdict['idpix_1b'] = idpix
            tdict['idwv_1b'] = idwave
        tdict['idpix'] = idpix
        tdict['idwv'] = idwave

        # HOLY 2
        tids = arholy.run_holy2(tcent, idpix, idwave, ameta['npix'], llist,
                                ngrid=ngrid, verbose=verbose)
                                #p23_frac=p23_frac, ngrid=ngrid, verbose=verbose)
        tdict['ids_2'] = tids
        # Evaluate
        tmsk, twv = set_ids_holy2(tcent, all_idpix, all_idwv)
        ID, gdID, badID = test_holy.evalaute_ids(tids, tmsk, twv)
        tdict['NID'] = np.sum(ID)
        tdict['NGOOD'] = np.sum(gdID)
        tdict['NBAD'] = np.sum(badID)
        # Append
        test_dicts.append(tdict.copy())
    # Output
    output_test_file(mdict, test_dicts, outfil)


# Command line execution
if __name__ == '__main__':
    print("SHOULD GENERATE A PARAM FILE FOR EACH TEST (YAML or JSON)")
    arc_file = 'LRISb_600.hdf5'
    outfil = 'output/LRISb_600_tests.hdf5'
    outfig = 'output/LRISb_600_tests.pdf'
    main(arc_file, 300, outfil, ngrid=800)
    fig_holytest(arc_file, outfil, outfig)