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


def init_holy2_test(infil, lamps=None):
    """
    Parameters
    ----------
    infil
    lamps

    Returns
    -------

    """
    if lamps is None:
        lamps=['ArI','NeI','HgI','KrI','XeI']
    # Open test solution
    root = '/Users/xavier/local/Python/PYPIT'
    wvsoln_fil = root+infil
    with open(wvsoln_fil) as data_file:
        wvsoln_dict = json.load(data_file)

    # Setup ID lines
    xnorm = (wvsoln_dict['xnorm']-1)
    npix = xnorm + 1  # ASSUMED
    all_idpix = np.array(wvsoln_dict['xfit'])*xnorm
    all_idwv = np.array(wvsoln_dict['yfit'])
    tcent = np.array(wvsoln_dict['tcent'])
    print('Number of Input IDs = {:d}'.format(len(all_idwv)))

    if False:
        func = 'polynomial'
        tmask, tparam = arutils.robust_polyfit(all_idwv, all_idpix, 4, function=func)
        tpixfit = arutils.func_val(tparam, all_idwv, func)
        print('tparam', tparam)
        trms = np.sqrt(np.mean((tpixfit-all_idpix)**2))
        print('full_RMS = {:g}'.format(trms))

    # Mask ID
    tmsk = tcent==tcent
    twv = np.zeros(len(tcent))
    for ii,itc in enumerate(tcent):
        if np.min(np.abs(itc-all_idpix)) > 0.1:
            tmsk[ii] = False
        else:
            twv[ii] = all_idwv[np.argmin(np.abs(itc-all_idpix))]

    # Linelist
    alist = ararclines.load_arcline_list(None,None, lamps, None)
    llist = np.array(alist['wave'])
    llist.sort()

    # Return
    return wvsoln_dict, npix, all_idpix, all_idwv, tcent, tmsk, twv, llist

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

def grab_id_lines(all_idpix, all_idwv, icen, nline):
    """
    Parameters
    ----------
    all_idpix
    all_idwv
    icen
    nline

    Returns
    -------

    """
    diff = np.abs(all_idpix-icen)
    asrt = np.argsort(diff)
    idpix = all_idpix[asrt[0:nline]]
    idpix.sort()
    idwave = all_idwv[asrt[0:nline]]
    idwave.sort()
    # Return
    return idpix, idwave

def evalaute_ids(tids, tmsk, twv):
    """
    Parameters
    ----------
    tids
    tmsk
    twv

    Returns
    -------

    """
    ID = tids[tmsk] > 1.
    gdID = np.abs(tids[tmsk]-twv[tmsk]) < 0.1
    badID = (tids[tmsk] > 1.) & (np.abs(tids[tmsk]-twv[tmsk]) > 0.1)
    #
    return ID, gdID, badID


def test_holy1(infil='/holy_grail/lrisr_600_7500_holy.json', verbose=False,
               outfil=None, ngrid=250, p23_frac = 0.25, lamps=None):
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
    #nlines = [3,4,5]
    nlines = [4,5]
    pixcen = np.round(np.linspace(100.,2000.,10)).astype(int)
    odict = dict(ngrid=ngrid, p23_frac=p23_frac, nlines=nlines, infil=infil,
                    pixcen=pixcen, runs={}, maxID=np.sum(tmsk))
    for nline in nlines:
        print('nline = {:d}'.format(nline))
        for icen in pixcen:
            idpix,idwave = grab_id_lines(all_idpix, all_idwv, icen, nline)
            # Extend?
            extend = False
            if extend:
                etids = arholy.extend_fit(tcent, idpix, idwave, llist)
                gdp = etids > 1.
                idpix = tcent[gdp]
                idwave = etids[gdp]
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
        #test_holy1(infil='lrisb_600_4000_holy.json',
        #           lamps=['ZnI', 'CdI', 'HgI'],
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
