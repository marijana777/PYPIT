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
              frac_extra=0.25, p23_frac=0.25, verbose=True,
              debug=False, close_tol=2.):
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

    Returns
    -------

    """
    from scipy.optimize import curve_fit

    # Fit ID lines with 2nd order polynomial
    func = 'polynomial'
    mask, pparam = arutils.robust_polyfit(idwave, idpix, 2, function=func)
    wmask, wparam = arutils.robust_polyfit(idpix, idwave, 2, function=func)
    pixfit = arutils.func_val(pparam, idwave, func)
    if verbose:
        prms = np.sqrt(np.mean((pixfit-idpix)**2))
        print('RMS = {:g}'.format(prms))

    # Setup global pixel fit
    wvmin, wvmax = np.min(idwave), np.max(idwave)
    wvcen = np.mean([wvmin,wvmax])
    pixcen = arutils.func_val(pparam, wvcen, func)
    dpixcen = pparam[1]*wvcen + pparam[2]*(wvmax**2-wvmin**2)/(wvmax-wvmin)*wvcen
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
    match_tol = 0.3  # Fraction of a pixel
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
        if (np.min(diff) < match_tol) & (nclose == 1):
            tids[kk] = llist[np.argmin(diff)]
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
    if debug:
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

