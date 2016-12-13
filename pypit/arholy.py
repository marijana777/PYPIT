from __future__ import (print_function, absolute_import, division, unicode_literals)

import numpy as np
from pypit import arpca
from pypit import armsgs
from pypit import arsave
from pypit import arutils
from pypit import ararc
from pypit.arplot import get_dimen as get_dimen
from pypit import ararclines
from pypit import arqa
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from sklearn.neighbors import KernelDensity
from collections import Counter
import scipy.interpolate as interpolate
from astropy.io import ascii
import os

try:
    from xastropy.xutils import xdebug as debugger
except:
    import pdb as debugger

# Logging
msgs = armsgs.get_logger()


def holy1(arc_spec, lamps, nsolsrch=10, numsearch=8, maxlin=0.2, npixcen=0.005, sig_rej=2.0):
    """ Automatically identify arc lines

    Parameters
    ----------
    lamps : array
      List of lamps that were turned on
    nsolsearch : int
      Number of candidate solutions to check
    numsearch : int
      Number of adjacent lines to use when deriving patterns
    maxlin : float 0-1
      The fraction of the detector that is considered linear
    npixcen : float
      The fraction of the detector that the central wavelength can be determined.
    sig_rej : float
      Rejection threshold (in units of 1 sigma) for the final fit

    Returns
    -------
    final_fit : dict
      Dictionary containing all relevant wavelength calibration information
    status : int
      The status of the fit
        0 = acceptable solution
        1 = marginally acceptable solution (0.1 < pixel RMS < 0.2)
        2 = unacceptable solution (pixel RMS > 0.2)
    """
    from pypit import arcyarc

    # Set the default status
    status = 0

    # Set the tolerance for acceptable identifications (in pixels)
    tolerance = 0.7

    # Get the number of pixels in the dispersion direction
    npixels = arc_spec.size
    # Convert npixcen to pixels
    npixcen *= npixels

    # Extract the arc
    #msgs.work("Detecting lines..")
    tampl, tcent, twid, w, satsnd, yprep = detect_lines()

    # Cut down to the good ones
    detlines = tcent[w]
    detampls = tampl[w]
    msgs.info('Detected {0:d} lines in the arc spectrum'.format(len(w[0])))

    # Sort the detected lines
    asrt = np.argsort(detlines)

    # Load the linelist
    #arcparam = setup_param(slf, sc, det, fitsdict)

    idx = slf._spect['arc']['index'][sc]
    disperser = fitsdict["disperser"][idx[0]]
    #lamps = slf._argflag['arc']['calibrate']['lamps']
    #lamps = arcparam['lamps']
    linelist = ararclines.load_arcline_list(slf, idx, lamps, disperser, wvmnx=None)
    ll = linelist['wave'].data
    whll = np.where(~ll.mask)
    ions = linelist['Ion'].data
    idsion = np.array(['     ']*whll[0].size)
    linelist = np.array(ll[whll])
    idsion[:] = np.array(ions[whll])
    # Sort the linelist
    linelist.sort()

    # Generate list patterns
    msgs.info("Generating linelist patterns")
    OutOfRange = 1.01*(np.max(linelist)-np.min(linelist))
    lstpatt, lstidx = arcyarc.patterns_sext(linelist, numsearch, OutOfRange)
    msgs.info("Number of reference patterns: {0:d}".format(lstidx.shape[0]))
    # Generate list tree
    msgs.info("Generating KDTree")
    lsttree = KDTree(lstpatt, leafsize=30)

    # Test both possibilities:
    # (tt=0) pixels correlate with wavelength, and
    # (tt=1) pixels anticorrelate with wavelength
    solscore = np.array([])
    solcorel = np.array([])
    solwaves = []
    solwvidx = []
    solwmask = []
    for tt in range(2):
        if tt == 0:
            msgs.info("Assuming pixels correlate with wavelength")
            detlines = detlines[asrt]
            detampls = detampls[asrt]
        else:
            msgs.info("Assuming pixels anticorrelate with wavelength")
            # Reverse the order of the pixels
            detlines = detlines[::-1]
            detampls = detampls[::-1]
            detlines = npixels - detlines
        # Generate the patterns and store indices
        detpatt, detidx = arcyarc.patterns_sext(detlines, numsearch, maxlin*npixels)
        msgs.info("Number of pixel patterns: {0:d}".format(detidx.shape[0]))

        if detidx.shape[0] == 0:
            status = 4
            msgs.info("Pixel patterns could not be generated")
            return None, status

        # Find the difference between the end point "reference" pixels
        ddiff = detlines[detidx[:, -1]] - detlines[detidx[:, 0]]

        # Create the KDTrees
        msgs.info("Creating KDTree of detected lines")
        dettree = KDTree(detpatt, leafsize=30)

        # Set the search error to be 1 pixel
        err = 2.0/np.min(ddiff)

        # Query the detections tree
        msgs.info("Cross-matching patterns")
        res = dettree.query_ball_tree(lsttree, r=err)

        # Assign wavelengths to each pixel
        msgs.info("Identifying candidate wavelengths")
        nrows = len(res)
        ncols = sum(map(len, res))
        if ncols == 0:
            status = 5
            msgs.info("Could not identify any matching patterns")
            return None, status
        nindx = detidx.shape[1]
        wvdisp = np.zeros(ncols)
        wvcent = np.zeros(ncols)
        wvindx = -1*np.ones((ncols*nindx, 3))
        cnt = 0
        for x in range(nrows):
            for y in range(len(res[x])):
                dx = detlines[detidx[x, -1]] - detlines[detidx[x, 0]]
                dp = linelist[lstidx[res[x][y], -1]] - linelist[lstidx[res[x][y], 0]]
                try:
                    null, cgrad = arutils.robust_polyfit(detlines[detidx[x, :]], linelist[lstidx[res[x][y], :]], 1,
                                                         sigma=2.0)
                    wvdisp[cnt] = cgrad[1]
                except:
                    wvdisp[cnt] = (dp/dx)
                coeff = arutils.func_fit(detlines[detidx[x, :]], linelist[lstidx[res[x][y]]], "polynomial", 2)
                wvcent[cnt] = arutils.func_val(coeff, npixels/2.0, "polynomial")
                for i in range(nindx):
                    wvindx[cnt*nindx+i, 0] = cnt
                    wvindx[cnt*nindx+i, 1] = detidx[x, i]
                    wvindx[cnt*nindx+i, 2] = lstidx[res[x][y], i]
                cnt += 1

        wvs = wvcent[np.where(wvcent != 0.0)]
        wvd = wvdisp[np.where(wvdisp != 0.0)]
        allwave = np.linspace(wvs.min(), wvs.max(), (np.max(linelist)-np.min(linelist))/wvd.min())
        bwest = npixcen*np.median(wvd)  # Assume the central wavelength can be determined within npixcen
        msgs.info("Constructing KDE with bandwidth {0:f}".format(bwest))
        wvkde = KernelDensity(bandwidth=bwest)
        wvkde.fit(wvs[:, np.newaxis])
        msgs.info("Constructing probability density function")
        wvpdf = np.exp(wvkde.score_samples(allwave[:, np.newaxis]))

        wvs.sort()
        msgs.info("Finding peaks of the probability density function")
        tpdf = wvpdf
        dtst = tpdf[1:]-tpdf[:-1]
        wmx = np.argwhere((dtst[1:] < 0.0) & (dtst[:-1] > 0.0)).flatten()
        msgs.info("Identified {0:d} peaks in the PDF".format(wmx.size))
        mxsrt = np.argsort(tpdf[wmx+1])[-nsolsrch:]
        # Estimate of the central wavelengths
        cwest = allwave[wmx+1][mxsrt][::-1]

        if True:
            cwv = np.linspace(0.0, 1.0, wvs.size)
            plt.subplot(211)
            plt.plot(wvs, cwv, 'b-')
            for i in range(cwest.size):
                plt.plot([cwest[i], cwest[i]], [0.0, 1.0], 'r-')
            plt.subplot(212)
            plt.hist(wvs, bins=np.linspace(wvs.min(), wvs.max(), 1500), normed=1)
            plt.plot(allwave, wvpdf, 'r-')
            plt.show()
            plt.clf()
            debugger.set_trace()

        tsolscore = np.ones(cwest.size)
        for i in range(cwest.size):
            msgs.info("Testing solution {0:d}/{1:d}".format(i+1, cwest.size))
            w = np.where((wvcent >= cwest[i]-bwest/2.0) & (wvcent <= cwest[i]+bwest/2.0))[0]
            # Only use potential solutions that have consistent dispersions
            try:
                msk, coeff = arutils.robust_polyfit(np.ones(w.size), wvdisp[w], 0, sigma=2.0)
                wm = np.where(msk == 0)
                if wm[0].size != 0:
                    w = w[wm]
            except:
                pass
            ind = np.in1d(wvindx[:, 0], w)
            detid = wvindx[ind, 1].astype(np.int)
            linid = linelist[wvindx[ind, 2].astype(np.int)]
            # Mask lines that deviate by at least 'tolerance' pixels from the best linear solution for each pattern
            mskid = arcyarc.find_linear(detlines[detid].reshape(-1, nindx), linid.reshape(-1, nindx), tolerance)
            ww = np.where(mskid.flatten() == 0)
            detid = detid[ww]
            linid = linid[ww]
            if ww[0].size == 0:
                tsolscore[i] = 0.0
                solwaves.append(None)
                solwmask.append(None)
                continue
            # Prepare the arrays used to store line identifications
            mskdone = np.zeros(detlines.size)
            wavdone = np.zeros(detlines.size)
            for j in range(detlines.size):
                # Find all ids for this line
                wdid = np.where(detid == j)
                lineid = linid[wdid]
                # Find most common wavelength id for this line
                if lineid.size != 0:
                    c = Counter(wv for wv in lineid)
                    comm = c.most_common(1)
                    wavdone[j] = comm[0][0]
                    mskdone[j] = 1
            # Now perform a brute force identification on the detlines and the linelist
            # Fit the known pixels with a cubic polynomial
            yval = detlines/(npixels-1.0)
            wmsk = np.where(mskdone == 1)

            # Determine (roughly) the minimum and maximum wavelengths
            coeff = arutils.func_fit(yval[wmsk], wavdone[wmsk], "polynomial", 1)
            wmnmx = arutils.func_val(coeff, np.array([0.0, 1.0]), "polynomial")
            wmin, wmax = wmnmx[0], wmnmx[1]

            wavmean = np.mean(wavdone[wmsk])
            xval = (wavdone - wavmean)/(wmax-wmin)
            ll = (linelist - wavmean)/(wmax-wmin)
            # Find the first and second coefficients of the polynomial fitting
            # These coefficients are given by (p0) the value of the cubic fit
            # at the mean wavelength of the id'ed pixels, and (p1) the value of
            # the derivative of the cubic fit at the mean wavelength of the id'ed
            # pixels.
            coeff = arutils.func_fit(xval[wmsk], yval[wmsk], "polynomial", 3)
            # ... and we only need the first two coefficients
            coeff = coeff[:2]

            msgs.info("Commencing brute force arc line identification")
            # Set a limit on the deviation of the final solution from linear
            # 0.3 = 30 per cent deviation from a linear solution
            lim = 0.3
            wdiff = (wmax-wmin)*lim
            wll = np.where((linelist > wmin-wdiff) & (linelist < wmax+wdiff))[0]
            wavidx = arcyarc.brute_force_solve(yval, ll[wll], coeff, npixels, lim)
            wavids = linelist[wll[wavidx]]
            mskids = np.zeros_like(wavids)
            mskids[np.where(wavidx == -1)] = 1
            # Fit the solution with a polynomial and repeat
            order = slf._argflag['arc']['calibrate']['polyorder']
            for rr in range(2):
                mskids, coeff = arutils.robust_polyfit(detlines, wavids, order, function="legendre",
                                                     initialmask=mskids, forceimask=True,
                                                     sigma=sig_rej, minv=0.0, maxv=npixels-1.0)
                model = arutils.func_val(coeff, detlines, "legendre", minv=0.0, maxv=npixels-1.0)
                # Calculate Angstroms per pixel at the location of each detection
                dcoeff = arutils.func_deriv(coeff, "legendre", 1)
                dmodel = arutils.func_val(dcoeff, detlines, "legendre", minv=0.0, maxv=npixels-1.0)
                wavids, wavidx = arcyarc.identify(linelist, model)
                mskids = np.zeros_like(wavids)
                mskids[np.where(np.abs((wavids-model)/dmodel) > tolerance)] = 1
            # Calculate the score for this solution
            # wgd = np.where(mskids == 0)
            # coeff = arutils.func_fit(detlines, wavids, "legendre", order, w=1-mskids, minv=0.0, maxv=npixels-1.0)
            # model = arutils.func_val(coeff, detlines, "legendre", minv=0.0, maxv=npixels-1.0)
            score = 1.0/np.std(wavids-model)
            tsolscore[i] = score

            # Store this solution for later comparison
            if tt == 0:
                solwaves.append(wavids.copy())
                solwvidx.append(wavidx.copy())
                solwmask.append(mskids.copy())
            else:
                solwaves.append(wavids.copy()[::-1])
                solwvidx.append(wavidx.copy()[::-1])
                solwmask.append(mskids.copy()[::-1])

        # Append the solution
        solscore = np.append(solscore, tsolscore)
        solcorel = np.append(solcorel, tt*np.ones_like(tsolscore))

    # Return detlines to its original order
    detlines = npixels - detlines
    detlines = detlines[::-1]

    # Identify the best solution
    scrbst = np.argmax(solscore)
    fmin, fmax = 0.0, 1.0
    ifit = np.where(solwmask[scrbst] == 0)
    xfit, yfit = detlines[ifit]/(npixels-1.0), solwaves[scrbst][ifit]
    mask, fit = arutils.robust_polyfit(xfit, yfit, slf._argflag['arc']['calibrate']['polyorder'],
                                       function="legendre", sigma=sig_rej, minv=fmin, maxv=fmax)
    irej = np.where(mask == 1)[0]
    if irej.size > 0:
        xrej = xfit[irej]
        yrej = yfit[irej]
        for imask in irej:
            msgs.info('Rejecting arc line {:g}'.format(yfit[imask]))
    else:
        xrej = []
        yrej = []
    xfit = xfit[mask == 0]
    yfit = yfit[mask == 0]
    # Get the name of the ions
    all_idsion = np.array(['12345']*len(tcent))
    all_idsion[ifit] = idsion[solwvidx[scrbst]]
    ions = all_idsion[ifit][mask == 0]

    dx = (1.0/npixels)**2
    cen_wd = arutils.func_val(fit, np.array([0.5, 0.5+dx]), "legendre", minv=fmin, maxv=fmax)
    cen_wave = cen_wd[0]
    cen_disp = np.abs(cen_wd[1]-cen_wd[0])/dx
    cen_disp /= (npixels-1.0)
    # Calculate the RMS
    model = arutils.func_val(fit, xfit, "legendre", minv=fmin, maxv=fmax)
    dmodel = np.abs(arutils.func_val(fit, xfit+dx, "legendre", minv=fmin, maxv=fmax))
    dmodel = (dmodel-model)/dx
    dmodel /= (npixels-1.0)
    pixrms = np.sqrt(np.sum(((yfit-model)/dmodel)**2)/xfit.size)
    tstc = arutils.func_fit(xfit, yfit, "polynomial", 4)
    tstp = arutils.func_val(tstc, xfit, "polynomial")
    np.std(tstp - xfit*(npixels-1.0))
    # Print the resulting properties of the best model
    msgs.info("Best wavlength solution is index {0:d} with estimated values:".format(scrbst) + msgs.newline() +
              "  Central wavelength: {0:.4f} Angstroms".format(cen_wave) + msgs.newline() +
              "  Central dispersion: {0:.4f} Angstroms/pixel".format(cen_disp) + msgs.newline() +
              "  RMS residuals: {0:.4f} pixels".format(pixrms))
    if solcorel[scrbst] == 0:
        msgs.info("Wavelength increases with increasing pixels")
    else:
        msgs.info("Wavelength decreases with increasing pixels")
    # Pack up fit
    final_fit = dict(fitc=fit, function="legendre", xfit=xfit, yfit=yfit,
                     ions=ions, fmin=fmin, fmax=fmax, xnorm=float(npixels),
                     xrej=xrej, yrej=yrej, mask=mask, spec=yprep, nrej=sig_rej,
                     shift=0.0)
    # QA
    arqa.arc_fit_qa(slf, final_fit)
    debugger.set_trace()
    # Return
    if pixrms > 0.3:
        msgs.warn("Pixel RMS from autoid exceeded an acceptable value of 0.3")
        msgs.warn("Wavelength solution is probably unreliable")
        status = 2
    elif pixrms > 0.15:
        msgs.warn("Pixel RMS is larger than ideal")
        msgs.info("Check the wavelength solution")
        status = 1
    else:
        msgs.info("Pixels residuals are acceptable")
    return final_fit, status


def detect_lines(censpec, MK_SATMASK=False):
    """
    Extract an arc down the center of the chip and identify
    statistically significant lines for analysis.

    Parameters
    ----------
    slf : Class instance
      An instance of the Science Exposure class
    det : int
      Index of the detector
    msarc : ndarray
      Calibration frame that will be used to identify slit traces (in most cases, the slit edge)
    censpec : ndarray, optional
      A 1D spectrum to be searched for significant detections
    MK_SATMASK : bool, optional
      Generate a mask of arc line saturation streaks? Mostly used for echelle data
      when saturation in one order can cause bleeding into a neighbouring order.

    Returns
    -------
    tampl : ndarray
      The amplitudes of the line detections
    tcent : ndarray
      The centroids of the line detections
    twid : ndarray
      The 1sigma Gaussian widths of the line detections
    w : ndarray
      An index array indicating which detections are the most reliable.
    satsnd : ndarray
      A mask indicating where which pixels contain saturation streaks
    yprep : ndarray
      The spectrum used to find detections. This spectrum has
      had any "continuum" emission subtracted off
    """
    from pypit import arcyarc
    # Extract a rough spectrum of the arc in each order
    msgs.info("Detecting lines")
    msgs.info("Extracting an approximate arc spectrum at the centre of the chip")
    if msgs._debug['flexure']:
        ordcen = slf._pixcen
    else:
        ordcen = slf.GetFrame(slf._pixcen, det)
    if censpec is None:
        #pixcen = np.arange(msarc.shape[slf._dispaxis], dtype=np.int)
        #ordcen = (msarc.shape[1-slf._dispaxis]/2)*np.ones(msarc.shape[slf._dispaxis],dtype=np.int)
        #if len(ordcen.shape) != 1: msgs.error("The function artrace.model_tilt should only be used for"+msgs.newline()+"a single spectrum (or order)")
        #ordcen = ordcen.reshape((ordcen.shape[0],1))
        msgs.work("No orders being masked at the moment")
        # Average over several pixels to remove some random fluctuations, and increase S/N
        op1 = ordcen+1
        op2 = ordcen+2
        om1 = ordcen-1
        om2 = ordcen-2
        censpec = (msarc[:,ordcen]+msarc[:,op1]+msarc[:,op2]+msarc[:,om1]+msarc[:,om2])/5.0
    # Generate a saturation mask
    if MK_SATMASK:
        ordwid = 0.5*np.abs(slf._lordloc[det-1] - slf._rordloc[det-1])
        msgs.info("Generating a mask of arc line saturation streaks")
        satmask = arcyarc.saturation_mask(msarc, slf._nonlinear[det-1])
        satsnd = arcyarc.order_saturation(satmask, ordcen, (ordwid+0.5).astype(np.int), slf._dispaxis)
    else:
        satsnd = np.zeros_like(ordcen)
    # Detect the location of the arc lines
    msgs.info("Detecting the strongest, nonsaturated lines")
    #####
    # Old algorithm for arc line detection
#   arcdet = arcyarc.detections_allorders(censpec, satsnd)
    #####
    # New algorithm for arc line detection
    #pixels=[]
    siglev = 6.0*slf._argflag['arc']['calibrate']['detection']
    bpfit = 5  # order of the polynomial used to fit the background 'continuum'
    fitp = slf._argflag['arc']['calibrate']['nfitpix']
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
    tpixt, num = arcyarc.detections_sigma(yprep, yerr, np.zeros(satsnd.shape[0], dtype=np.int), siglev/2.0, siglev)
    pixt = arcyarc.remove_similar(tpixt, num)
    pixt = pixt[np.where(pixt != -1)].astype(np.int)
    tampl, tcent, twid, ngood = arcyarc.fit_arcorder(xrng, yprep, pixt, fitp)
    w = np.where((np.isnan(twid) == False) & (twid > 0.0) & (twid < 10.0/2.35) & (tcent > 0.0) & (tcent < xrng[-1]))
    # Check the results
    #plt.clf()
    #plt.plot(xrng,yprep,'k-')
    #plt.plot(tcent,tampl,'ro')
    #plt.show()
    # Return
    return tampl, tcent, twid, w, satsnd, yprep
