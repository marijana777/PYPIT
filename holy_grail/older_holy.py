
#############################################################
#############################################################
#############################################################
#############################################################

def simple_match_lines(guesses, llist, toler=0.5):
    """ Match a set of input lines to a given line list
    Simple for loop.  Ok for ~10 guesses

    Parameters
    ----------
    guesses : ndarray
      (assumed Ang)
    llist : ndarray
    toler : float
      Matching tolerance.  Assumed Ang

    Returns
    -------
    matches : ndarray
      Set to 0. for no match within tolerance
    """
    # Dumb approach for now
    matches = np.zeros_like(guesses)
    for ii,guess in enumerate(guesses):
        diff = np.abs(guess-llist)
        gdm = np.where(diff < toler)[0]
        if len(gdm) == 0:
            continue
        elif len(gdm) == 1:
            matches[ii] = llist[gdm[0]]
        else:
            try:
                imin = np.argmin(diff[gdm])
            except ValueError:
                debugger.set_trace()
            matches[ii] = llist[gdm[imin]]
    # Return
    return matches


def extrap_Nlines(lines_pix, ids_idx, ids_wave, llist, fdict=None, Nextrap=5):
    """ Extrapolate the fit by N lines on each side (as possible)

    Parameters
    ----------
    lines_pix
    ids_idx
    ids_wave
    llist
    fdict
    Nextrap

    Returns
    -------

    """
    dwvdpix = (ids_wave[1]-ids_wave[0])/(
        lines_pix[ids_idx[1]]-lines_pix[ids_idx[0]])
    nlines = len(lines_pix)
    if fdict is None:
        raise ValueError("Not ready for this.  Provide a fit dict")
    # Grab Nextrap lines on each side
    ids_min = np.min(ids_idx)
    ids_max = np.max(ids_idx)
    ilow = max(0,ids_min-Nextrap)
    ihi = min(ids_max+Nextrap+1,nlines)

    # Guess wave
    extrap_lines = np.concatenate([lines_pix[ilow:ids_min],
                                  lines_pix[ids_max+1:ihi]])
    #xv = 2.0 * (extrap_lines-fdict['xmin'])/(fdict['xmax']-fdict['xmin']) - 1.0
    guesses = arutils.func_val(fdict['coeff'], extrap_lines, fdict['func'],
                               minv=fdict['minv'],maxv=fdict['maxv'])
    # Match
    matches = simple_match_lines(guesses, llist, toler=dwvdpix*2.)
    #debugger.set_trace()

    return extrap_lines, guesses, matches

def searching_for_the_grail(pixpk=None, idlines=None, npix=2048,
                            llist=None, extrap_off=500.):
    """ ID lines in quadrants to solve the Holy Grail

    Returns
    -------

    """
    # Peaks
    if pixpk is None:
        pixpk = init_pixpk()
    # Input IDs
    if idlines is None:
        idlines = init_idlines()
    # Linelist
    if llist is None:
        llist = ararclines.load_arcline_list(None,None,
                                         ['ArI','NeI','HgI','KrI','XeI'],None)

    # Interpolate first [not yet implemented]

    # Go lower first
    go_lower = True
    while go_lower:
        minID = idlines.min_pix
        pix0 = minID - extrap_off

        # Fit IDs [using them all for the moment]
        idlines.set_all()
        idlines.fit_poly(2)
        allpk = pixpk < minID

        # Extrap
        go_lower = extrap_lines(idlines, pixpk[allpk], llist,
                                min_extrap=11, pix0=pix0)
        # ID lines within current IDs (*aggressive* rejection)
        identify_newlines(idlines, pixpk, llist)

    msgs.info("Extrapolating to higher lines")
    go_higher = True
    while go_higher:
        maxID = idlines.max_pix
        pix1 = maxID + extrap_off

        # Fit IDs [using them all for the moment]
        idlines.set_closest(maxID, 5)
        idlines.fit_poly(2)
        #debugger.set_trace()
        allpk = pixpk > maxID

        # Extrap
        go_higher = extrap_lines(idlines, pixpk[allpk], llist,
                                min_extrap=11, pix1=pix1)#, debug=True)
        if go_higher is False:
            go_higher = extrap_lines(idlines, pixpk[allpk], llist,
                                     min_extrap=9, pix1=pix1, tolerpix=4.0,
                                     debug=True)
        # ID lines within current IDs (*aggressive* rejection)
        identify_newlines(idlines, pixpk, llist)

    # Final fit
    idlines.set_all()
    idlines.fit_poly(3)
    identify_newlines(idlines, pixpk, llist, do_inbetween=False)#, debug=True)
    idlines.set_all()
    idlines.fit_poly(3)
    waves = idlines.eval_poly(idlines.all_pix)
    rms = np.sqrt(np.sum((waves-idlines.all_wave)**2)/idlines.npix)
    print('rms = {:g}'.format(rms))
    debugger.set_trace()

def identify_newlines(idlines, pixpk, llist, do_inbetween=True,
                      debug=False):
    """
    Parameters
    ----------
    idlines
    pixpk

    Returns
    -------

    """
    # Fit all ID lines
    idlines.set_all()
    idlines.fit_poly(3)
    xc = idlines.curr_fit['xc']
    #  Possible ones
    if do_inbetween:
        inbetween = np.where((pixpk>idlines.min_pix) & (pixpk<idlines.max_pix))[0]
        if len(inbetween) == 0:
            msgs.info("No lines avaialble for fitting inbetween the ID lines")
        #  No duplicates
        pixpk = pixpk[inbetween]
    msk = pixpk == pixpk
    for ii,pix in enumerate(pixpk):
        if idlines.in_allpix(pix):
            msk[ii] = False
    #  Assign fit wavelengths
    waves = idlines.eval_poly(pixpk)
    # Search for a strict match
    matches = simple_match_lines(waves, llist['wave'], toler=idlines.dwv*0.3) # strict
    badm = matches < 1
    msk[badm] = False
    gd_try = np.sum(msk)
    if gd_try == 0:
        msgs.info("No lines available for fitting")
    # Fit to mask
    fitmsk = np.where(msk)[0]
    pix = np.concatenate([idlines.pix, pixpk[fitmsk]])
    wave = np.concatenate([idlines.wave, matches[fitmsk]])
    mask, coeff = arutils.robust_polyfit(pix-xc, wave, 3, sigma=2.)
    # Final mask
    rej = np.where(mask[idlines.npix:])[0]
    msk[fitmsk[rej]] = False
    if debug:
        debugger.set_trace()
    # Add to idlines
    for idx in np.where(msk)[0]:
        idlines.add_ID(pixpk[idx], matches[idx])

def init_idlines():
    """
    Returns
    -------

    """
    id_dict = {'dlamb': 1.6,
               'first_id_idx': [30, 35, 37, 43, 44],
               'first_id_pix': np.array([ 638.93076161,  746.09652748,  770.25572854,  902.1922555 , 924.47068438]),
               'first_id_wave': np.array([ 6508.326 ,  6680.1201,  6718.8974 ,  6931.3788,  6967.352]),
               }
    #'first_id_wave': np.array([ 6506.528 ,  6678.2766,  6717.043 ,  6929.4672,  6965.431 ]),
    idlines = IDLines(id_dict['first_id_pix'], id_dict['first_id_wave'])
    return idlines


def extrap_lines(idlines, pixpk, llist, dwvdpix=None, tolerpix=4.,
                 pix0=None, pix1=None, min_extrap=11, min_nhits=9,
                 debug=True):
    """
    Parameters
    ----------
    idlines
    pixpk
    llist
    dwvdpix
    tolerpix

    Returns
    -------

    """
    import matplotlib.pyplot as plt
    coeff = idlines.curr_fit['coeff']
    xc = idlines.curr_fit['xc']
    #
    if dwvdpix is None:
        dwvdpix = idlines.dwv
    # Convert to wavelengths
    wavepk = arutils.func_val(idlines.curr_fit['coeff'], pixpk-xc,
                               idlines.curr_fit['func'],
                               minv=idlines.curr_fit['minv'],
                               maxv=idlines.curr_fit['maxv'])
    # Match
    matches = simple_match_lines(wavepk, llist['wave'], toler=dwvdpix*tolerpix)
    gdm = np.where(matches > 1.)[0]
    if len(gdm) < min_extrap:
        msgs.warn("Not enough matches to extrapolate ({:d})".format(len(gdm)))
        #debugger.set_trace()
        return False
    # Cut
    if pix0 is not None: # low sid3
        gdpk = np.where(pixpk[gdm] > pix0)[0]
        if len(gdpk) < min_extrap:
            gdm = gdm[-11:]
            debugger.set_trace()
        else:
            msgs.info("Cut a few lines to limit extrapolation")
            gdm = gdm[gdpk]
    if pix1 is not None: # high side
        gdpk = np.where(pixpk[gdm] < pix1)[0]
        if len(gdpk) < min_extrap:
            gdm = gdm[:11]
        else:
            msgs.info("Cut a few lines to limit extrapolation")
            gdm = gdm[gdpk]

    # Loop on matches
    lpix = list(idlines.pix)
    lwave = list(idlines.wave)
    orig = False
    if orig:
        all_pix_norm = np.concatenate([pixpk[gdm],idlines.pix]) - xc
        all_wave = np.concatenate([matches[gdm],idlines.wave])
    else:
        all_pix_norm = pixpk[gdm] - xc
        all_wave = matches[gdm]
    hits = np.zeros_like(all_pix_norm).astype(int)
    nhits = np.zeros(len(gdm)).astype(int)
    thresh = dwvdpix/4.
    for jj,igdm in enumerate(gdm):
        # Fit
        coeffr = arutils.func_fit(np.array(lpix+[pixpk[igdm]])-xc,
                                  np.array(lwave+[matches[igdm]]),
                                  'polynomial', 3)
        #debugger.set_trace()
        # Calculate at every pix including idlines
        fit_wave = arutils.func_val(coeffr, all_pix_norm, 'polynomial')
        # Thresh
        hit = np.abs(fit_wave-all_wave) < thresh
        hits[hit] += 1
        nhits[jj] = np.sum(hit)
        #print(jj, nhits[jj])

    # Is nhits > min_nhits?
    if np.max(nhits) > min_nhits:
        msgs.info("Exceed minimum hits for large extrapolation [good]")
        mxhit = np.max(nhits)
        mxpk = np.where(nhits == mxhit)[0]
        dx = pixpk[gdm[mxpk]]-xc
        imatch = gdm[mxpk[np.argmax(np.abs(dx))]]
    else:
        msgs.info("Did not exceed minimum hits for large extrapolation [bad]")
        msgs.info("Going convservative..")
        # Take the highest hit and the farthest as the ID
        mxhit = np.max(hits)
        mxpk = np.where(hits == mxhit)[0]
        dx = pixpk[gdm[mxpk]]-xc
        imatch = gdm[mxpk[np.argmax(np.abs(dx))]]
        #debugger.set_trace()

    if debug:
        coeffr = arutils.func_fit(idlines.pix-xc,
                                  idlines.wave, 'polynomial', 1)
        Dlp = (matches[gdm]-coeffr[0]-coeffr[1]*(pixpk[gdm]-xc))/(pixpk[gdm]-xc)**2
        plt.clf()
        ax = plt.gca()
        #ax.scatter(ex_lines[gdp]-xmean, matches[gdp]-guesses[gdp], marker='o')
        dx = pixpk[gdm]-xc
        ax.scatter(dx, Dlp, marker='o')
        ax.set_xlabel('dx')
        ax.set_ylabel('dlambda_prime')
        ax.set_ylim(-3e-5,7e-5)
        plt.show()
        plt.close()
        #debugger.set_trace()

    #
    idlines.add_ID(pixpk[imatch], matches[imatch])
    msgs.info("Added ID {:g} at wavelength {:g} with {:d} hits".format(
            pixpk[imatch], matches[imatch], mxhit))


    return True

    #



    """
    # 2 parameter poly fit
    coeffr = arutils.func_fit(idlines.pix-xc,
                              idlines.wave, 'polynomial', 1)

    Dlp = (matches[gdm]-coeffr[0]-coeffr[1]*(pixpk[gdm]-xc))/(pixpk[gdm]-xc)**2

    # Fit
    mask, coeff_Dlp = arutils.robust_polyfit(pixpk[gdm]-xc, Dlp, 1, sigma=2.)

    #Dlp_ids = (idlines.wave-coeffr[0]-coeffr[1]*(idlines.pix-xc))/(idlines.pix-xc)**2
    debug = True
    if debug:
        plt.clf()
        ax = plt.gca()
        #ax.scatter(ex_lines[gdp]-xmean, matches[gdp]-guesses[gdp], marker='o')
        dx = pixpk[gdm]-xc
        ax.scatter(dx, Dlp, marker='o')
        #dx = idlines.pix-xc
        #ax.scatter(dx, Dlp_ids, marker='s', color='red')
        xval = np.linspace(-1000., 0., 1000)
        yfit = arutils.func_val(coeff_Dlp, xval, func='polynomial')
        ax.plot(xval, yfit, 'r')
        ax.set_xlabel('dx')
        ax.set_ylabel('dlambda_prime')
        ax.set_ylim(-3e-5,7e-5)
        plt.show()
        debugger.set_trace()

    # Next term
    Dlp_best = np.median(Dlp)
    Dlpp = (matches[gdm]-coeffr[0]-coeffr[1]*(pixpk[gdm]-xc)-
        Dlp_best*(pixpk[gdm]-xc)**2)/(pixpk[gdm]-xc)**3
    if debug:
        plt.clf()
        ax = plt.gca()
        dx = pixpk[gdm]-xc
        ax.scatter(dx, Dlpp, marker='o')
        ax.set_xlabel('dx')
        ax.set_ylabel('dlambda_primeprime')
        ax.set_ylim(-4e-7,3e-7)
        plt.show()
        np.median(Dlpp)
        debugger.set_trace()
    Dlpp_best = np.median(Dlpp)
    wave_xx = (coeffr[0] + coeffr[1]*(pixpk[gdm]-xc) +
               Dlp_best*(pixpk[gdm]-xc)**2 + Dlpp_best*(pixpk[gdm]-xc)**3)
    if debug:
        # Plot
        plt.clf()
        ax = plt.gca()
        gdp = matches > 1.
        #ax.scatter(ex_lines[gdp]-xmean, matches[gdp]-guesses[gdp], marker='o')
        dx = pixpk[gdm]-xc
        ax.scatter(dx, (matches[gdp]-wave_xx), marker='o')
        ax.set_xlabel('dx')
        ax.set_ylabel('dlambda')
        #ax.set_ylim(-0.01,0.01)
        plt.show()
    """

def init_pixpk():
    pixpk = np.array([44.48548296,   172.89127284,   186.40595723,   194.90669321,
         205.27791998,   225.91831224,   237.75445526,   244.67787642,
         258.32024281,   284.80424904,   297.95088588,   304.32791548,
         338.93921337,   367.06152237,   380.85908214,   401.33172725,
         410.50564368,   423.48900142,   434.99430789,   457.37290774,
         488.37853661,   499.77642185,   512.47037332,   526.46054509,
         531.09374127,   548.60194487,   561.55045742,   584.23322514,
         607.38695571,   616.29664611,   638.93076161,   655.40375678,
         669.6613697 ,   696.61898826,   730.00992529,   746.09652748,
         759.13834741,   770.25572854,   792.32336615,   866.2691866 ,
         873.17420623,   882.14114981,   888.34678116,   902.1922555 ,
         924.47068438,   961.43743412,   976.85878178,   987.46838199,
        1020.05739901,  1036.93430325,  1053.50315   ,  1085.1768635 ,
        1114.722588  ,  1122.68159311,  1140.78101461,  1154.30159287,
        1159.95601591,  1176.25448288,  1183.35928052,  1217.26159628,
        1237.96559256,  1248.14347416,  1257.39153162,  1264.07798538,
        1277.12260434,  1309.02417381,  1369.4388766 ,  1375.18929295,
        1407.1859496 ,  1441.21856771,  1474.36353717,  1494.81185267,
        1510.61385111,  1520.04403868,  1532.17586078,  1544.00013256,
        1553.49714119,  1568.13495337,  1601.20827538,  1615.38852212,
        1648.8922041 ,  1682.28030842,  1692.21254363,  1738.46846657,
        1773.96586387,  1779.93551015,  1786.2796231 ,  1791.82491476,
        1799.09112006,  1852.68590012,  1866.14740391,  1872.67491904,
        1881.15295659,  1889.01947025,  1903.32590177,  1913.46875905,
        1920.45244571,  1927.45068105,  1932.74364454,  1941.98544882,
        1952.75881331,  1960.00649953,  1969.44860805,  1980.91775367,
        1989.21778856,  1998.94533717,  2025.86827108])
    return pixpk


class IDLines(object):

    def __init__(self, pix=None, wave=None):

        # pix and wave
        if pix is not None:
            self.all_pix = pix
            self.orig_pix = pix.copy()
            self.pix = pix.copy()
        else:
            self.all_pix = None
            self.orig_pix = None
            self.pix = None
        if wave is not None:
            self.all_wave = wave
            self.orig_wave = wave.copy()
            self.wave = wave.copy()
        else:
            self.all_wave = None
            self.orig_wave = None
            self.wave = None
        # Fit
        self.curr_fit = {}

    @property
    def min_pix(self):
        return np.min(self.all_pix)

    @property
    def max_pix(self):
        return np.max(self.all_pix)

    @property
    def npix(self):
        return self.all_pix.size

    @ property
    def dwv(self):
        xc = self.curr_fit['xc']
        wvs = arutils.func_val(self.curr_fit['coeff'],
                               np.array([xc,xc+1]),
                               self.curr_fit['func'],
                               minv=self.curr_fit['minv'],
                               maxv=self.curr_fit['maxv'])
        dwvdpix = np.abs(wvs[1]-wvs[0])
        return dwvdpix



    def add_ID(self, pix, wave):
        """ Add a new ID'd line
        Parameters
        ----------
        pix
        wave

        Returns
        -------

        """
        min_diff = np.min(np.abs(pix-self.all_pix))
        if min_diff < 1.:
            raise IOError("ID already exists")
        # Add
        self.all_pix = np.concatenate([self.all_pix, np.array([pix])])
        isrt = np.argsort(self.all_pix)
        self.all_pix = self.all_pix[isrt]
        self.all_wave = np.concatenate([self.all_wave, np.array([wave])])[isrt]


    def eval_poly(self, pixels):
        """
        Parameters
        ----------
        pixels

        Returns
        -------

        """
        waves = arutils.func_val(self.curr_fit['coeff'],
                               pixels-self.curr_fit['xc'],
                               self.curr_fit['func'],
                               minv=self.curr_fit['minv'],
                               maxv=self.curr_fit['maxv'])
        #
        return waves

    def fit_poly(self, deg):
        """
        Parameters
        ----------
        deg

        Returns
        -------

        """
        self.curr_fit['func'] = 'polynomial'
        self.curr_fit['deg'] = deg
        self.curr_fit['minv'] = None
        self.curr_fit['maxv'] = None
        self.curr_fit['xc'] = np.mean(self.pix)
        self.curr_fit['coeff'] = arutils.func_fit(self.pix-self.curr_fit['xc'],
                                                     self.wave,
                                                     self.curr_fit['func'],
                                                     self.curr_fit['deg'])

    def in_allpix(self, pix):
        """
        Parameters
        ----------
        pix

        Returns
        -------

        """
        if np.min(np.abs(self.all_pix-pix)) < 0.3:
            return True
        else:
            return False

    def set_all(self):
        """ Set pix, wave to the all values
        """
        self.pix = self.all_pix.copy()
        self.wave = self.all_wave.copy()

    def set_closest(self, pix, nclosest):
        """
        """
        closest = np.argsort(np.abs(self.all_pix-pix))
        idx = closest[0:nclosest]
        self.pix = self.all_pix[idx]
        self.wave = self.all_wave[idx]
        # Sort
        spix = np.argsort(self.pix)
        self.pix = self.pix[spix]
        self.wave = self.wave[spix]

"""
    # Setup quadrants
    nreg = 4
    pix_reg = npix//nreg
    regs = np.zeros((nreg,2)).astype(int)
    regs[:,0] = np.arange(nreg)*pix_reg
    regs[:,1] = (np.arange(nreg)+1)*pix_reg
    regs[-1,1] = min(npix,regs[-1,1])

    nlin = np.zeros(nreg).astype(int)
    for reg in range(nreg):
        pixin = np.where((idlines.orig_pix>=regs[reg,0]) &
                         (idlines.orig_pix<regs[reg,1]))[0]
        nlin[reg] = len(pixin)

    # Interpolate here

    #

    # Work on existing quadrant(s)
    # Only the highest/lowest will be un-populated
    idx = np.where(nlin>0)[0]
    i0 = np.min(idx)-1
    i1 = np.max(idx)+1
    pix0 = regs[i0,0]
    pix1 = regs[i1,1]
"""