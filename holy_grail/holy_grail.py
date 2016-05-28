# Tests for the Holy Grail
from __future__ import print_function, absolute_import, division, unicode_literals
import linetools.utils as ltu

import sys, os, json
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

try:
    from xastropy.xutils import xdebug as debugger
except:
    import pdb as debugger


def generate_noncalib(tcent, all_idpix, wvsoln_dict):
    """ Generate a list of noncalib parameters
    Parameters
    ----------
    tcent
    all_idpix
    wvsoln_dict

    Returns
    -------

    """
    msktc = tcent == tcent
    for jj,ipix in enumerate(all_idpix):
        diff = np.abs(tcent-ipix)
        if np.min(diff) < 1.:
            msktc[np.argmin(diff)] = False
    newtc = tcent[msktc]
    newwv = arutils.func_val(wvsoln_dict['fitc'], newtc/(wvsoln_dict['xnorm']-1),
                             wvsoln_dict['function'],
                             minv=wvsoln_dict['fmin'], maxv=wvsoln_dict['fmax'])
    # Return
    return newwv


def test_holy1(infil='/holy_grail/lrisr_600_7500_holy.json', verbose=False,
               outfil=None, ngrid=250, p23_frac = 0.25, lamps=None,
               nlines=(3,4,5)):
    """ Test number and location of Holy 1 lines

    Parameters
    ----------
    infil : str, optional
      JSON file containing the good fit

    Returns
    -------

    """
    # Init
    wvsoln_dict, npix, all_idpix, all_idwv, tcent, tmsk, twv, llist = init_holy2_test(infil, lamps=lamps)

    # Loop on nlines and pixcen
    pixcen = np.round(np.linspace(100.,2000.,10)).astype(int)
    odict = dict(ngrid=ngrid, p23_frac=p23_frac, nlines=nlines, infil=infil,
                    pixcen=pixcen, runs={}, maxID=np.sum(tmsk))
    for nline in nlines:
        print('nline = {:d}'.format(nline))
        for icen in pixcen:
            idpix,idwave = grab_id_lines(all_idpix, all_idwv, icen, nline)
            # Extend?
            if nline < 4:
                idpix,idwave = arholy.add_lines(tcent, idpix, idwave, llist, verbose=verbose)
            extend = False
            if extend:
                idpix,idwave = arholy.extend_fit(tcent, idpix, idwave, llist, match_toler=0.3, extrap=1)
            # Holy2
            tids = arholy.run_holy2(tcent, idpix, idwave, npix, llist,
                             p23_frac=p23_frac, ngrid=ngrid, verbose=verbose)
            # Evaluate
            ID, gdID, badID = evalaute_ids(tids, tmsk, twv)
            # Fill
            key = '{:d}_{:d}'.format(nline,icen)
            odict['runs'][key] = {}
            odict['runs'][key]['NID'] = np.sum(ID)
            odict['runs'][key]['NGOOD'] = np.sum(gdID)
            odict['runs'][key]['NBAD'] = np.sum(badID)
            if verbose:
                print('NGOOD={:d}, NBAD={:d}'.format(np.sum(gdID),np.sum(badID)))

    # Write?
    if outfil is not None:
        gddict = ltu.jsonify(odict)
        ltu.savejson(outfil, gddict, overwrite=True)
        print("Wrote output to {:s}".format(outfil))


def test_tcent(infil='/holy_grail/lrisr_600_7500_holy.json',
               outfil=None, ngrid=250, p23_frac = 0.25,
               nmodify=None,
               seed=1234):
    """  Test adding false tcent or removing real ones
    Parameters
    ----------
    infil
    outfil
    ngrid
    p23_frac

    Returns
    -------

    """
    # Init
    wvsoln_dict, npix, all_idpix, all_idwv, tcent, tmsk, twv, llist = init_holy2_test(infil)
    pixcen = 1024
    nline = 4
    ntrial = 5
    ntcent = len(tcent)
    rstate = np.random.RandomState(seed)

    # Loop on modifying tcent
    if nmodify is None:
        nmodify = [-20, -10, 10, 40]
    #pixcen = np.round(np.linspace(100., 2000., 3)).astype(int)
    pixcen = np.round(np.linspace(100., 2000., 5)).astype(int)
    odict = dict(ngrid=ngrid, p23_frac=p23_frac, nlines=nline, infil=infil,
                    pixcen=pixcen, nmodify=nmodify, ntrial=ntrial, runs={})
    for nmod in nmodify:
        print('nmodify = {:d}'.format(nmod))
        for icen in pixcen:
            idpix, idwave = grab_id_lines(all_idpix, all_idwv, icen, nline)
            # Trials
            for qq in xrange(ntrial):
                if nmod < 0:
                    msk = tcent == tcent
                    rand = rstate.random_sample(ntcent)
                    srt = np.argsort(rand)
                    msk[srt[0:-1*nmod]] = False
                    mtcent = tcent[msk]
                    mtmsk = tmsk[msk]
                    mtwv = twv[msk]
                elif nmod > 0:
                    newtc = npix*rstate.random_sample(nmod)
                    mtcent = np.concatenate([tcent,newtc])
                    mtmsk = np.concatenate([tmsk, np.array([False]*nmod)])
                    mtwv = np.concatenate([twv, np.zeros(nmod)])
                # Run
                tids = arholy.run_holy2(mtcent, idpix, idwave, npix, llist,
                                 p23_frac=p23_frac, ngrid=ngrid, verbose=False)
                # Evaluate
                ID, gdID, badID = evalaute_ids(tids, mtmsk, mtwv)
                # Fill
                key = '{:d}_{:d}_{:d}'.format(icen, nmod, qq)
                odict['runs'][key] = {}
                odict['runs'][key]['NID'] = np.sum(ID)
                odict['runs'][key]['NGOOD'] = np.sum(gdID)
                odict['runs'][key]['NBAD'] = np.sum(badID)
                odict['runs'][key]['maxID'] = np.sum(mtmsk)
    # Write?
    if outfil is not None:
        gddict = ltu.jsonify(odict)
        ltu.savejson(outfil, gddict, overwrite=True)
        print("Wrote output to {:s}".format(outfil))


""" PLOTS """
def plot_ngoodbad(json_fil, parms, title, outfil, lgd_loc='lower left'):
    """
    Parameters
    ----------
    json_fil
    title
    xlbl
    outfil

    Returns
    -------

    """
    odict = ltu.loadjson(json_fil)
    all_runs = odict['runs'].keys()
    nruns = len(all_runs)
    # Parameters
    nparm = len(parms)
    if nparm != len(all_runs[0].split('_')):
        raise ValueError("Parameter coding is not as expected!")
    if nparm > 3:
        raise ValueError("Not ready for this")

    # Start the plot
    pp = PdfPages(outfil)

    plt.figure(figsize=(5, 5))
    plt.clf()
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    lsz = 15.
    ax.set_ylabel('N_GOOD', fontsize=lsz)

    ax2 = ax.twinx()
    ax2.set_ylabel('N_BAD', fontsize=lsz)

    # Nparam
    symbols  = ['o', 's', 'v', 'x', '^']
    if nparm >= 2:
        # Points use parm0, x is from parm1
        # Parse runs
        parm0 = np.zeros(nruns).astype(int)
        parm1 = np.zeros(nruns).astype(int)
        ngood = np.zeros(nruns)
        nbad = np.zeros(nruns)
        for kk,run in enumerate(all_runs):
            vals = run.split('_')
            parm0[kk] = int(vals[0])
            parm1[kk] = int(vals[1])
            ngood[kk] = odict['runs'][run]['NGOOD']
            nbad[kk] = odict['runs'][run]['NBAD']
        for ss, iparm0 in enumerate(odict[parms[0]]):
            gd0 = np.where(parm0 == iparm0)[0]
            ax.scatter(parm1[gd0], ngood[gd0], marker=symbols[ss], color='blue',
                       label='{}'.format(iparm0))
            ax2.scatter(parm1[gd0], nbad[gd0], marker=symbols[ss], color='red')

    ax.set_xlabel(parms[1], size=lsz)

    # Legend
    #legend = ax.legend(loc='lower left', borderpad=0.3,
    #                   handletextpad=0.3, fontsize='large')
    legend = ax.legend(loc=lgd_loc, scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='small', numpoints=1)

    # Label
    #ax.text(0.05, 0.87, lbl, transform=ax.transAxes, color='black', size=csz, ha='left')
    #lbl2 = '({:d} systems)'.format(nsys)
    #ax.text(0.05, 0.75, lbl2, transform=ax.transAxes, color='black', size=csz, ha='left')
    # Axes
    #ax.xaxis.set_major_locator(plt.MultipleLocator(1.))

    plt.title(title, fontsize=20)
    # End
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig()
    pp.close()
    plt.close()



#################
def main(flg_test):

    if flg_test == 'all':
        flg_test = np.sum( np.array( [2**ii for ii in range(15)] ))
    else:
        flg_test = int(flg_test)

    # Fiducial
    if (flg_test % 2**1) >= 2**0:
        test_lrisr_600_7500()

    # Holy2 with varying Holy1 input
    if (flg_test % 2**2) >= 2**1:
        test_holy1(infil='/holy_grail/lrisr_600_7500_holy.json',
                   lamps=['ArI','NeI','HgI','KrI','XeI'],
                   ngrid=500, outfil='test_holy1_lrisr600_500.json')
        plot_ngoodbad('test_holy1_lrisr600_500.json', ['nlines', 'pixcen'],
                      'Testing Holy1 Input (LRISr 600)', 'test_holy1_lrisr600.pdf')
        #
        test_holy1(infil='/holy_grail/lrisb_600_4000_holy.json',
                   lamps=['ZnI', 'CdI', 'HgI'], p23_frac=0.4,
                   ngrid=500, outfil='test_holy1_lrisb600.json')
        #test_holy1(infil='lrisb_600_4000_holy.json',
        #           lamps=['ZnI', 'CdI', 'HgI'], nlines=[4,5],
        #           ngrid=250, outfil='test_holy1_lrisb600_500.json')

    # Holy2 tcent
    if (flg_test % 2**3) >= 2**2:
        test_tcent(infil='lrisr_600_7500_holy.json', ngrid=500, outfil='test_tcent_lrisr600.json')

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_test = 0
        #flg_test += 2**0   # LRISr
        flg_test += 2**1   # Holy1
    else:
        flg_fig = sys.argv[1]

    main(flg_test)
