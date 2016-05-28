import numpy as np
import json


import armsgs
import arutils
import ararclines

try:
    from xastropy.xutils import xdebug as debugger
except:
    import pdb as debugger

# Logging
msgs = armsgs.get_logger()


def run_holy2(tcent, idpix, idwave, npix, llist, noncalib=None, ngrid=100,
              frac_extra=0.25, p23_frac=0.5, verbose=True,
              debug=False, close_tol=2., ndeg=2, match_tol=0.6):
    """ Run Holy 2

    Parameters
    ----------
    tcent
    idpix
    idwave
    llist
    noncalib
    frac_extra : float, optinal
      Fraction of estimated spectral range to expand linelist matching
    p23_frac : float, optional
      Fraction of detector to allows shift from p2 and p3
    close_tol : float, optional
      If there are multiple lines within close_tol, don't ID this line
    match_tol : float, optional

    Returns
    -------

    """
    #from scipy.optimize import curve_fit

    # Fit ID lines with 2nd order polynomial
    func = 'polynomial'
    mask, pparam = arutils.robust_polyfit(idwave, idpix, ndeg, function=func)
    wmask, wparam = arutils.robust_polyfit(idpix, idwave, ndeg, function=func)
    pixfit = arutils.func_val(pparam, idwave, func)
    if verbose:
        prms = np.sqrt(np.mean((pixfit-idpix)**2))
        print('RMS = {:g}'.format(prms))
        print('pparam', pparam)
        tmask, tparam = arutils.robust_polyfit(idwave, idpix, 3, function=func)
        print('tparam', tparam)
        tpixfit = arutils.func_val(tparam, idwave, func)
        trms = np.sqrt(np.mean((tpixfit-idpix)**2))
        print('tRMS = {:g}'.format(trms))

    # Setup global pixel fit
    wvmin, wvmax = np.min(idwave), np.max(idwave)
    wvcen = np.mean([wvmin,wvmax])
    pixcen = arutils.func_val(pparam, wvcen, func)
    if ndeg == 1:
        dpixcen = pparam[1]*wvcen
    elif ndeg == 2:
        dpixcen = pparam[1]*wvcen + pparam[2]*(wvmax**2-wvmin**2)/(wvmax-wvmin)*wvcen
    else:
        raise ValueError("Not ready for this")
    chk = False
    if chk:
        wvval = (idwave - wvcen)/wvcen
        idpval = idpix - pixcen - dpixcen*wvval
        debugger.xpcol(wvval, idpix, idpval)


    '''
    # Right answer
    def x2x3_fit(x, p2, p3):
        return p2*x**2 + p3*x**3
    xxval = (all_idwv-wvcen)/wvcen
    yyval = all_idpix-pixcen-dpixcen*xxval
    ppopt, ppcov = curve_fit(x2x3_fit, xxval, yyval)

    ends = arutils.func_val(wvsoln_dict['fitc'], np.array([0.,1.]), wvsoln_dict['function'],
                            minv=wvsoln_dict['fmin'], maxv=wvsoln_dict['fmax'])
    '''

    # Cut down line list
    dwv = wparam[1]
    extra = frac_extra * npix * dwv
    ends = arutils.func_val(wparam, np.array([0.,npix]), func)
    keep = np.where((llist > ends[0]-extra) & (llist < ends[1]+extra))[0]
    llist = llist[keep]

    # Add in non-calib lines
    if noncalib is not None:
        allwv = np.concatenate([llist,noncalib])
    else:
        allwv = llist

    # Setup xval
    allwv.sort()
    xall = (allwv-wvcen)/wvcen
    xall2 = xall**2
    xall3 = xall**3

    # Setup p2 and p3 limits
    xmx = np.max(np.abs(xall))
    p2min = -1 * p23_frac * npix / xmx**2
    p2max = p23_frac * npix / xmx**2
    p3min = -1 * p23_frac * npix / xmx**3
    p3max = p23_frac * npix / xmx**3

    # TODO
    #  Scan from +/- npix in p2 and p3
    #  Should ignore lines with 2 matches within 1 pix

    # Generate pix image
    pix_img = np.outer(tcent, np.ones(len(allwv)))

    # Ready to go
    '''  1D (p3 only)
    tst_val = np.linspace(0., 2*ppopt[1], ntst)
    tst_metric = np.zeros(ntst)
    wv_to_pix_setup = pixcen + dpixcen*xall + ppopt[0]*xall2
    for jj in xrange(ntst):
        # The next line could be 'mapped'
        wv_to_pix = wv_to_pix_setup + tst_val[jj]*xall3
        # Main call
        tst_metric[jj] = holy_cross_lines(pix_img, wv_to_pix)
        if (jj % 10) == 0:
            print('jj = {:d}'.format(jj))
    '''
    #tst_val2 = np.linspace(0., 2*ppopt[0], ngrid)
    #tst_val3 = np.linspace(0., 2*ppopt[1], ngrid)
    scan_p2 = np.linspace(p2min, p2max, ngrid)
    scan_p3 = np.linspace(p3min, p3max, ngrid)
    metric = np.zeros((ngrid,ngrid))
    wv_to_pix_setup = pixcen + dpixcen*xall
    for ii in xrange(ngrid):
        for jj in xrange(ngrid):
            # The next lines could/should be 'mapped'
            wv_to_pix = wv_to_pix_setup + scan_p2[ii]*xall2 + scan_p3[jj]*xall3
            # Main call
            metric[ii,jj] = holy_cross_lines(pix_img, wv_to_pix)
        #if (ii % 10) == 0:
        #    print('ii = {:d}'.format(ii))
    if verbose:
        print('max = {:g}'.format(np.max(metric)))

    # ID lines (Line list only)
    min_idx = np.where(metric == np.max(metric))
    match_p2 = scan_p2[min_idx[0][0]]
    match_p3 = scan_p3[min_idx[1][0]]
    if verbose:
        print('p2 = {:g}'.format(match_p2))
        print('p3 = {:g}'.format(match_p3))
    xmatch = (llist-wvcen)/wvcen
    match_pix = pixcen + dpixcen*xmatch + match_p2*xmatch**2 + match_p3*xmatch**3
    tids = np.zeros(len(tcent))
    for kk,ipix in enumerate(tcent):
        diff = np.abs(match_pix-ipix)
        nclose = np.sum(diff < close_tol)
        #if verbose:
        #    print('min for {:g} is {:g}'.format(ipix, np.min(diff)))
        if (np.min(diff) < match_tol):
            if nclose == 1:
                tids[kk] = llist[np.argmin(diff)]
            elif verbose:
                print('2 close lines')
    # Test again input
    '''
    aids = np.zeros(len(all_idpix))
    for kk,ipix in enumerate(all_idpix):
        diff = np.abs(match_pix-ipix)
        if np.min(diff) < match_tol:
            aids[kk] = llist[np.argmin(diff)]
    debugger.xpcol(all_idpix, aids, aids-all_idwv)
    '''
    if verbose:
        print('NID = {:d}'.format(np.sum(tids > 1)))
    #debugger.xpcol(tcent, tids)#, aids-all_idwv)
    #debug=True
    if debug:
        # Plot 1D
        if False:
            from matplotlib import pyplot as plt
            plt.clf()
            ax = plt.gca()
            ax.scatter(tcent, [0.5]*len(tcent), label='tcent')
            for mpix in match_pix:
                ax.plot([mpix]*2, (0,1), 'r')
            plt.show()
            plt.close()
        #
        debugger.set_trace()
        debugger.ximshow(metric)
    # Return
    return tids


def holy_cross_lines(pix_img, wv_to_pix, max_off=5., two_inv_sigma_sq=2./4):
    """
    Parameters
    ----------
    pix
    wv
    soln
    two_inv_sigma_sq : float
      Inverse of sigma**2 * 2

    Returns
    -------
    chi2 : float

    """
    global all_idpix

    # Generate wavelength image
    wv_img = np.outer(np.ones(pix_img.shape[0]), wv_to_pix)
    # Take difference
    diff_img = np.abs(pix_img-wv_img)
    min_diff = np.min(diff_img, axis=1)
    keep = min_diff < max_off
    # Calculate
    metric = np.sum(np.exp(-1*min_diff[keep]**2/two_inv_sigma_sq))
    # Return
    return metric


def extend_fit(tcent, idpix, idwv, llist, match_toler=1., extrap=1):
    """ Same underlying algorithm for extending the fit as in ararc.simple_calib
    Parameters
    ----------
    tcent
    idpix
    idwv
    llist
    match_toler

    Returns
    -------

    """
    # Init
    min_in, max_in = np.min(idwv), np.max(idwv)
    nid = len(idwv)
    # Indices for initial fit
    ifit = []
    for ipix in idpix:
        ifit.append(np.argmin(np.abs(ipix-tcent)))
    ifit = np.array(ifit)
    # Setup for fitting
    sv_ifit = list(ifit) # Keep the originals
    all_ids = -999.*np.ones(len(tcent))
    #all_idsion = np.array(['12345']*len(tcent))
    all_ids[ifit] = idwv
    #all_idsion[ifit] = idsion[gd_str]
    # Fit
    n_order = min(2, nid-2)
    n_final = 3
    func = 'polynomial'
    flg_quit = False
    nsig_rej = 3.
    fmin, fmax = -1., 1.

    # Now extrapolate
    while (n_order <= n_final) and (flg_quit is False):
        # Fit with rejection
        xfit, yfit = tcent[ifit], all_ids[ifit]
        mask, fit = arutils.robust_polyfit(xfit, yfit, n_order, function=func, sigma=nsig_rej, minv=fmin, maxv=fmax)
        # Reject but keep originals (until final fit)
        ifit = list(ifit[mask == 0]) + sv_ifit
        # Find new points (should we allow removal of the originals?)
        twave = arutils.func_val(fit, tcent, func, minv=fmin, maxv=fmax)
        for ss,iwave in enumerate(twave):
            mn = np.min(np.abs(iwave-llist))
            if mn/fit[1] < match_toler:
                imn = np.argmin(np.abs(iwave-llist))
                # Update and append
                all_ids[ss] = llist[imn]
                #all_idsion[ss] = llist['Ion'][imn]
                ifit.append(ss)
        # Keep unique ones
        debugger.set_trace()
        ifit = np.unique(np.array(ifit,dtype=int))
        # Increment order
        if n_order < n_final:
            n_order += 1
        else:
            # This does 2 iterations at the final order
            flg_quit = True
    # Interp if possible
    debugger.set_trace()
    # Return
    return all_ids


def add_lines(itcent, idpix, idwv, llist, itoler=2., nextrap=1, verbose=False):
    """Attempt to identify and add additional goodlines

    Parameters
    ----------
    id_dict : dict
      dict of ID info
    pixpk : ndarray
      Pixel locations of detected arc lines
    toler : float
      Tolerance for a match (pixels)
    gd_lines : ndarray
      array of expected arc lines to be detected and identified
    inpoly : int, optional
      Order of polynomial for fitting for initial set of lines

    Returns
    -------
    id_dict : dict
      Filled with complete set of IDs and the final polynomial fit
    """
    # Insure these are sorted
    tcent = itcent.copy()
    tcent.sort()
    minp_in, maxp_in = np.min(idpix), np.max(idpix)
    func = 'polynomial'
    nsig_rej = 2.5
    fmin, fmax = -1., 1.

    # Indices for initial IDs
    ifit = []
    for ipix in idpix:
        ifit.append(np.argmin(np.abs(ipix-tcent)))
    ifit = np.array(ifit)
    if len(ifit) < 3:
        raise ValueError("This is not a good idea..")

    # TODO
    #  First try to interpolate to add 1 or more lines

    # Now try to extrapolate
    # TODO
    #  First try to interpolate to add 1 or more lines
    all_ids = -999.*np.ones(len(tcent))
    all_ids[ifit] = idwv
    n_order = min(2, len(idwv)-2)
    xfit, yfit = tcent[ifit], all_ids[ifit]
    mask, fit = arutils.robust_polyfit(xfit, yfit, n_order, function=func, sigma=nsig_rej, minv=fmin, maxv=fmax)
    # Edges
    ilow = np.min(ifit)
    ihi = np.max(ifit)
    pos=True

    # Loop on additional lines for identification
    nsuccess = 0
    allfit = list(ifit)
    toler = itoler
    while nsuccess < nextrap:
        # index to add (step on each side)
        if pos:
            ihi += 1
            inew = ihi
            pos=False
        else:
            ilow -= 1
            inew = ilow
            pos=True
        if (ilow < 0) & (ihi > (tcent.size-1)):
            if verbose:
                print("Not enough matches with tolerance={:g}. Doubling".format(toler))
            toler *= 2.
            if toler > 10.:
                debugger.set_trace()
                print("I give up")
                break
            allfit = list(ifit)
            ilow = np.min(ifit)
            ihi = np.max(ifit)
            pos=True
            continue
        if (inew < 0) or (inew > (tcent.size-1)):  # Off ends
            continue
        # New line
        new_pix = tcent[inew]
        # newwv
        newwv = arutils.func_val(fit, new_pix, func, minv=fmin, maxv=fmax)
        # Match
        mnm = np.min(np.abs(llist-newwv))
        if mnm > np.abs(toler*fit[1]):
            print("No match for {:g}, mnm={:g}".format(new_pix, mnm/fit[1]))
            continue
        # TODO
        # Make sure there are not two lines close by here

        # REFIT and check RMS
        chkifit = np.array(allfit + [inew])
        imin = np.argmin(np.abs(llist-newwv))
        all_ids[inew] = llist[imin]
        xfit, yfit = tcent[chkifit], all_ids[chkifit]
        mask, chkfit = arutils.robust_polyfit(xfit, yfit, 2, function=func, sigma=nsig_rej, minv=fmin, maxv=fmax)
        wvfit = arutils.func_val(chkfit, xfit, func)
        rms = np.sqrt(np.mean((wvfit-yfit)**2))
        if verbose:
            print('RMS = {:g}'.format(rms/fit[1]))
        if rms < 0.1:
            if verbose:
                print('Added {:g} at {:g}'.format(llist[imin], new_pix))
            allfit += [inew]
        elif verbose:
            print("Failed RMS test")
        nsuccess += 1
    #
    new_idpix = tcent[allfit]
    new_idwv = all_ids[allfit]
    # Return
    return new_idpix, new_idwv
