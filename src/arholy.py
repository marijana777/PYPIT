import numpy as np
import armsgs
import arutils
import ararclines

try:
    from xastropy.xutils import xdebug as debugger
except:
    import pdb as debugger

# Logging
msgs = armsgs.get_logger()


def init_holy2():
    """ Initialize the start of holy phase 2
    Returns
    -------

    """
    import json
    from scipy.optimize import curve_fit

    # Open test solution
    root = '/Users/xavier/local/Python/PYPIT'
    wvsoln_fil = root+'/test_suite/lrisr_600_7500_holy.json'
    with open(wvsoln_fil) as data_file:
        wvsoln_dict = json.load(data_file)
    xnorm = (wvsoln_dict['xnorm']-1)
    all_idpix = np.array(wvsoln_dict['xfit'])*xnorm
    all_idwv = np.array(wvsoln_dict['yfit'])

    # Grab inner 5 lines (as our initial test)
    diff = np.abs(all_idpix-1024)
    asrt = np.argsort(diff)
    idpix = all_idpix[asrt[0:5]]
    idpix.sort()
    idwave = all_idwv[asrt[0:5]]
    idwave.sort()

    # Fit with 3rd order
    func = 'polynomial'
    mask, pparam = arutils.robust_polyfit(idwave, idpix, 2, function=func)
    pixfit = arutils.func_val(pparam, idwave, func)
    prms = np.sqrt(np.mean((pixfit-idpix)**2))
    print('RMS = {:g}'.format(prms))

    # Setup global pixel fit
    wvmin, wvmax = np.min(idwave), np.max(idwave)
    wvcen = np.mean([wvmin,wvmax])
    pixcen = arutils.func_val(pparam, wvcen, func)
    dpixcen = pparam[1]*wvcen + pparam[2]*(wvmax**2-wvmin**2)/(wvmax-wvmin)*wvcen
    chk = True
    if chk:
        wvval = (idwave - wvcen)/wvcen
        idpval = idpix - pixcen - dpixcen*wvval
        debugger.xpcol(wvval, idpix, idpval)

    def x2x3_fit(x, p2, p3):
        return p2*x**2 + p3*x**3

    # Right answer
    xxval = (all_idwv-wvcen)/wvcen
    yyval = all_idpix-pixcen-dpixcen*xxval
    ppopt, ppcov = curve_fit(x2x3_fit, xxval, yyval)

    # Linelist
    alist = ararclines.load_arcline_list(None,None,['ArI','NeI','HgI','KrI','XeI'],None)
    llist = np.array(alist['wave'])
    llist.sort()
    ends = arutils.func_val(wvsoln_dict['fitc'], np.array([0.,1.]), wvsoln_dict['function'],
                            minv=wvsoln_dict['fmin'], maxv=wvsoln_dict['fmax'])
    keep = np.where((llist > ends[0]-200) & (llist < ends[1]+200))[0]
    llist = llist[keep]

    # Add in non-calib lines
    tcent = np.array(wvsoln_dict['tcent'])
    msktc = tcent == tcent
    for jj,ipix in enumerate(all_idpix):
        diff = np.abs(tcent-ipix)
        if np.min(diff) < 1.:
            msktc[np.argmin(diff)] = False
    newtc = tcent[msktc]
    newwv = arutils.func_val(wvsoln_dict['fitc'], newtc/xnorm, wvsoln_dict['function'],
                             minv=wvsoln_dict['fmin'], maxv=wvsoln_dict['fmax'])
    allwv = np.concatenate([llist,newwv])
    allwv.sort()
    xall = (allwv-wvcen)/wvcen
    xall2 = xall**2
    xall3 = xall**3

    # TODO
    #  Scan from +/- npix in p2 and p3

    # Generate pix image
    pix_img = np.outer(all_idpix, np.ones(len(allwv)))

    # Ready to go
    ntst = 100
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
    debugger.set_trace()

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
