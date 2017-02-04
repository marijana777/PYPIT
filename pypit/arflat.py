from __future__ import (print_function, absolute_import, division, unicode_literals)

import numpy as np
from pypit import arcomb
from pypit import armsgs
from pypit import arutils
from pypit import arparse as settings
from pypit import arqa
from pypit import arpca

from pypit import ardebug as debugger

# Logging
msgs = armsgs.get_logger()


def flatfield(slf, sciframe, flatframe, det, snframe=None,
              varframe=None):
    """ Flat field the input image
    Parameters
    ----------
    slf
    sciframe : 2d image
    flatframe : 2d image
    snframe : 2d image, optional
    det : int
      Detector index

    Returns
    -------
    flat-field image
    and updated sigma array if snframe is input
    or updated variance array if varframe is input

    """
    if (varframe is not None) & (snframe is not None):
        msgs.error("Cannot set both varframe and snframe")
    # New image
    retframe = np.zeros_like(sciframe)
    w = np.where(flatframe > 0.0)
    retframe[w] = sciframe[w]/flatframe[w]
    if w[0].size != flatframe.size:
        ww = np.where(flatframe <= 0.0)
        slf._bpix[det-1][ww] = 1.0
    # Variance?
    if varframe is not None:
        retvar = np.zeros_like(sciframe)
        retvar[w] = varframe[w]/flatframe[w]**2
        return retframe, retvar
    # Error image
    if snframe is None:
        return retframe
    else:
        errframe = np.zeros_like(sciframe)
        wnz = np.where(snframe>0.0)
        errframe[wnz] = retframe[wnz]/snframe[wnz]
        return retframe, errframe


def flatnorm(slf, det, msflat, maskval=-999999.9, overpix=6, plotdesc=""):
    """ Normalize the flat-field frame

    Parameters
    ----------
    slf : class
      An instance of the Science Exposure class
    det : int
      Detector number
    msflat : ndarray
      Flat-field image
    maskval : float
      Global floating point mask value used throughout the code
    overpix : int
      overpix/2 = the number of pixels to extend beyond each side of the order trace
    plotdesc : str
      A title for the plotted QA

    Returns
    -------
    msnormflat : ndarray
      The normalized flat-field frame
    msblaze : ndarray
      A 2d array containing the blaze function for each slit
    """
    from pypit import arcyutils
    from pypit import arcyextract
    from pypit import arcyproc
    dnum = settings.get_dnum(det)

    msgs.info("Normalizing the master flat field frame")
    norders = slf._lordloc[det-1].shape[1]
    # First, determine the relative scale of each amplifier (assume amplifier 1 has a scale of 1.0)
    if (settings.spect[dnum]['numamplifiers'] > 1) & (norders > 1):
        sclframe = get_ampscale(slf, det, msflat)
        # Divide the master flat by the relative scale frame
        msflat /= sclframe
    # Determine the blaze
    polyord_blz = 2  # This probably doesn't need to be a parameter that can be set by the user
    # Look at the end corners of the detector to get detector size in the dispersion direction
    #xstr = slf._pixlocn[det-1][0,0,0]-slf._pixlocn[det-1][0,0,2]/2.0
    #xfin = slf._pixlocn[det-1][-1,-1,0]+slf._pixlocn[det-1][-1,-1,2]/2.0
    #xint = slf._pixlocn[det-1][:,0,0]
    # Find which pixels are within the order edges
    msgs.info("Identifying pixels within each order")
    ordpix = arcyutils.order_pixels(slf._pixlocn[det-1], slf._lordloc[det-1], slf._rordloc[det-1])
    msgs.info("Applying bad pixel mask")
    ordpix *= (1-slf._bpix[det-1].astype(np.int))
    mskord = np.zeros(msflat.shape)
    msgs.info("Rectifying the orders to estimate the background locations")
    #badorders = np.zeros(norders)
    msnormflat = maskval*np.ones_like(msflat)
    msblaze = maskval*np.ones((msflat.shape[0],norders))
    msgs.work("Must consider different amplifiers when normalizing and determining the blaze function")
    msgs.work("Multiprocess this step to make it faster")
    flat_ext1d = maskval*np.ones((msflat.shape[0],norders))
    for o in range(norders):
        # Rectify this order
        recframe = arcyextract.rectify(msflat, ordpix, slf._pixcen[det-1][:,o], slf._lordpix[det-1][:,o],
                                       slf._rordpix[det-1][:,o], slf._pixwid[det-1][o]+overpix, maskval)
        if settings.argflag["reduce"]["flatfield"]["method"].lower() == "bspline":
            msgs.info("Deriving blaze function of slit {0:d} with a bspline".format(o+1))
            tilts = slf._tilts[det - 1].copy()
            gdp = (msflat != maskval) & (ordpix == o + 1)
            srt = np.argsort(tilts[gdp])
            everyn = settings.argflag['reduce']['flatfield']['params'][0]
            if everyn > 0.0 and everyn < 1.0:
                everyn *= msflat.shape[0]
                everyn = int(everyn + 0.5)
            everyn *= slf._pixwid[det - 1][o]
            bspl = arutils.func_fit(tilts[gdp][srt], msflat[gdp][srt], 'bspline', 3, everyn=everyn)
            model_flat = arutils.func_val(bspl, tilts.flatten(), 'bspline')
            model = model_flat.reshape(tilts.shape)
            word = np.where(ordpix == o + 1)
            msnormflat[word] = msflat[word] / model[word]
            msblaze[:, o] = arutils.func_val(bspl, np.linspace(0.0, 1.0, msflat.shape[0]), 'bspline')
            mskord[word] = 1.0
            flat_ext1d[:, o] = np.sum(msflat * mskord, axis=1) / np.sum(mskord, axis=1)
            mskord *= 0.0
        elif settings.argflag["reduce"]["flatfield"]["method"].lower() == "polyscan":
            polyorder = settings.argflag["reduce"]["flatfield"]["params"][0]
            polypoints = settings.argflag["reduce"]["flatfield"]["params"][1]
            repeat = settings.argflag["reduce"]["flatfield"]["params"][2]
            # Take the median along the spatial dimension
            flatmed = np.median(recframe, axis=1)
            # Perform a polynomial fitting scheme to determine the blaze profile
            xarray = np.arange(flatmed.size, dtype=np.float)
            weight = flatmed.copy()
            msgs.work("Routine doesn't support user parameters yet")
            msgs.bug("Routine doesn't support user parameters yet")
            blazet = arcyutils.polyfit_scan(xarray, flatmed.copy(), weight, maskval, polyorder, polypoints, repeat)
             # Remove the masked endpoints
            outx, outy, outm, lox, hix = arcyproc.remove_maskedends(xarray, flatmed, blazet, maskval)
            # Inspect the end points and extrapolate from the best fitting end pixels
            derv = (outm[1:]-outm[:-1])/(outx[1:]-outx[:-1])
            dervx = 0.5*(outx[1:]+outx[:-1])
            derv2 = (derv[1:]-derv[:-1])/(dervx[1:]-dervx[:-1])
            medv = np.median(derv2)
            madv = 1.4826*np.median(np.abs(derv2-medv))
            blaze = arcyproc.blaze_fitends(outx, outy, outm, derv2-medv, madv, polyord_blz, polypoints)
            #plt.plot(xarray,flatmed,'k-',drawstyle='steps')
            #plt.plot(xarray, blaze, 'r-')
            #plt.show()
            #np.savetxt("check_blaze_ord{0:d}.txt".format(o),np.transpose((xarray,flatmed)))
            # Divide the flat by the fitted flat profile
            finalblaze = np.ones(recframe.shape[0])
            finalblaze[lox:hix] = blaze.copy()
            blazenrm = finalblaze.reshape((finalblaze.size, 1)).repeat(recframe.shape[1], axis=1)
            recframe /= blazenrm
            # Store the blaze for this order
            msblaze[lox:hix,o] = blaze.copy()
            flat_ext1d[:,o] = flatmed.copy()
            # Sort the normalized frames along the dispersion direction
            recsort = np.sort(recframe, axis=0)
            # Find the mean value, but only consider the "innermost" 50 per cent of pixels (i.e. the pixels closest to 1.0)
            recmean = arcyproc.scale_blaze(recsort, maskval)
            #rows = np.arange(recsort.shape[0]/4,(3*recsort.shape[0])/4,dtype=np.int)
            #w = np.ix_(rows,np.arange(recframe.shape[1]))
            #recmean = np.mean(recsort[w],axis=0)
            for i in range(recmean.size):
                recframe[:, i] /= recmean[i]
            # Undo the rectification
            normflat_unrec = arcyextract.rectify_undo(recframe, slf._pixcen[det-1][:,o], slf._lordpix[det-1][:,o],
                                                      slf._rordpix[det-1][:,o], slf._pixwid[det-1][o], maskval,
                                                      msflat.shape[0], msflat.shape[1])
            # Apply the normalized flatfield for this order to the master normalized frame
            msnormflat = arcyproc.combine_nrmflat(msnormflat, normflat_unrec, slf._pixcen[det-1][:,o],
                                                  slf._lordpix[det-1][:,o], slf._rordpix[det-1][:,o],
                                                  slf._pixwid[det-1][o]+overpix, maskval)
        else:
            msgs.error("Flatfield method {0:s} is not supported".format(settings.argflag["reduce"]["flatfield"]["method"]))
    # Send the blaze away to be plotted and saved
    if "2dpca" in settings.argflag["reduce"]["flatfield"].keys():
        if settings.argflag["reduce"]["flatfield"]["2dpca"] >= 1:
            msgs.info("Performing a 2D PCA on the blaze fits")
            msblaze = arpca.pca2d(msblaze, settings.argflag["reduce"]["flatfield"]["2dpca"])
    # Plot the blaze model
    msgs.info("Saving blaze fits to QA")
    arqa.plot_orderfits(slf, msblaze, flat_ext1d, desc=plotdesc, textplt="Order")
    # If there is more than 1 amplifier, apply the scale between amplifiers to the normalized flat
    if (settings.spect[dnum]['numamplifiers'] > 1) & (norders > 1):
        msnormflat *= sclframe
    return msnormflat, msblaze


def get_ampscale(slf, det, msflat):
    """ Normalize the flat-field frame

    Parameters
    ----------
    slf : class
      An instance of the Science Exposure class
    det : int
      Detector number
    msflat : ndarray
      Flat-field image

    Returns
    -------
    sclframe : ndarray
      A frame to scale all amplifiers to the same counts at the amplifier borders
    """
    dnum = settings.get_dnum(det)

    sclframe = np.ones_like(msflat)
    ampdone = np.zeros(settings.spect[dnum]['numamplifiers'], dtype=int) # 1 = amplifiers have been assigned a scale
    ampdone[0]=1
    while np.sum(ampdone) != settings.spect[dnum]['numamplifiers']:
        abst, bbst, nbst, n0bst, n1bst = -1, -1, -1, -1, -1 # Reset the values for the most overlapping amplifier
        for a in range(0, settings.spect[dnum]['numamplifiers']): # amplifier 'a' is always the reference amplifier
            if ampdone[a] == 0: continue
            for b in range(0, settings.spect[dnum]['numamplifiers']):
                if ampdone[b] == 1 or a == b: continue
                tstframe = np.zeros_like(msflat)
                tstframe[np.where(slf._datasec[det-1] == a+1)] = 1
                tstframe[np.where(slf._datasec[det-1] == b+1)] = 2
                # Determine the total number of adjacent edges between amplifiers a and b
                n0 = np.sum(tstframe[1:,:]-tstframe[:-1,:])
                n1 = np.sum(tstframe[:,1:]-tstframe[:,:-1])
                if (abs(n0)+abs(n1)) > nbst:
                    n0bst = n0
                    n1bst = n1
                    nbst = abs(n0)+abs(n1)
                    abst = a
                    bbst = b
        # Determine the scaling factor for these two amplifiers
        tstframe = np.zeros_like(msflat)
        tstframe[np.where(slf._datasec[det-1] == abst+1)] = 1
        tstframe[np.where(slf._datasec[det-1] == bbst+1)] = 2
        if abs(n0bst) > abs(n1bst):
            # The amplifiers overlap on the zeroth index
            w = np.where(tstframe[1:,:]-tstframe[:-1,:] != 0)
            sclval = np.median(msflat[w[0][0]+1, w[1]])/np.median(msflat[w[0][0], w[1]])
            # msflat[w[0][0], w[1][0:50]] = 1.0E10
            # msflat[w[0][0]-1, w[1][0:50]] = -1.0E10
            # arutils.ds9plot(msflat)
            if n0bst > 0:
                # Then pixel w[0][0] falls on amplifier a
                sclval = sclframe[w[0][0], w[1]] * sclval
            else:
                # pixel w[0][0] falls on amplifier b
                sclval = sclframe[w[0][0]+1, w[1]] / sclval
        else:
            # The amplifiers overlap on the first index
            w = np.where(tstframe[:,1:]-tstframe[:,:-1] != 0)
            sclval = np.median(msflat[w[0], w[1][0]+1]/msflat[w[0], w[1][0]])
            if n1bst > 0:
                # Then pixel w[1][0] falls on amplifier a
                sclval = sclframe[w[0], w[1][0]] * sclval
            else:
                # pixel w[1][0] falls on amplifier b
                sclval = sclframe[w[0], w[1][0]+1] / sclval
        # Finally, apply the scale factor thwe amplifier b
        w = np.where(slf._datasec[det-1] == bbst+1)
        sclframe[w] = np.median(sclval)
        ampdone[bbst] = 1
    return sclframe

def slitless(fitsdict, setup_dict, npoly1=7):

    from pypit import arload
    from pypit import arproc

    sless_files = []
    n_sless = len(sless_files)
    if (n_sless <= 2): sigrej_pixel = 1.0
    elif (n_sless == 3): sigrej_pixel = 1.1
    elif (n_sless == 4): sigrej_pixel = 1.3
    elif (n_sless == 5): sigrej_pixel = 1.6
    elif (n_sless == 6): sigrej_pixel = 1.9
    else: sigrej_pixel = 2.0

    # Data sections
    # Bias
    settings.argflag['bias']['useframe'] = 'overscan'

    # Stack images first (why did we not do this in LowRedux??)
    ind = np.arange(len(fitsdict['filename']))
    for kk in range(settings.spect['mosaic']['ndet']):
        det = kk + 1  # Detectors indexed from 1
        arproc.get_datasec_trimmed(None, fitsdict, det, 0)
        frames = arload.load_frames(fitsdict, ind, det, frametype='pixel flat', msbias='overscan')
        # Scale
        nframe = frames.shape[2]
        for ii in range(1,nframe):
            scale = np.median(frames[:,:,ii]/frames[:,:,0])
            frames[:,:,ii] /= scale
        # Combine
        msslessflat = arcomb.comb_frames(frames, det, 'slitless', printtype='slitless flat')

        # Now the magic from LowRedux
        nx, ny = msslessflat.shape
        piximg = np.outer(np.ones(nx), np.arange(ny)) + np.outer((np.arange(nx)/float(nx-1)/10.), np.ones(ny))
        waveimg = piximg + 1000.
        igood = np.where(msslessflat > 2000.)[0]  # THIS IS RISKY
        ngood = np.sum(igood)

        npercol = int(float(ngood) / ny) > 1
        npoly = min(npoly1, int(npercol / 10.))
        npoly = max(npoly,1)

        # Sort the pixels in wavelength
        ii = igood[np.argsort(piximg[igood])]
        #; Always use the same breakpoints
        ybkpt = 0

        invvar = nframe*(msslessflat>0.)/(msslessflat + (msslessflat==0))  # Reduce variance by number of frames
        tempivar = invvar[igood]
        long_flatfield_specillum, piximg[igood], ximg[igood] $
                                        , image1[igood], spec_set $
                                        , illum_set $
                                        , SLITSAMP=slitsamp $
                                        , invvar = tempivar $
                                        , slitwidth = midwidth[slitid-1] $
                                        , modelfit = modelfit $
                                        , FINEBKPT=kast $
                                        , ybkpt = ybkpt $
                                        , npoly = npoly, CHK = CHK $
                                        , PIXFIT = (use_pixel[ifile] EQ 1)