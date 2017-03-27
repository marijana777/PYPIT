from __future__ import division
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy import interpolate
from astropy.io import fits
import matplotlib.pylab as plt
import os
import seaborn as sns
from scipy.interpolate import interp1d
sns.axes_style("darkgrid")

def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2.0 / (2 * c**2))
    
def GausSlitLoss(width,height,xcenter=0,ycenter=0):
    #width and height should be in the same units, preferably in units of FWHM
    psf = np.array([1.000,   .995,   .985,   .971,   .954, .933,   .911,   .886,   .860,   .833, .804,   .774,   .743,   .713,   .682, .651,   .620,   .594,   .559,   .529, .500,   .471,   .443,   .417,   .391, .366,   .342,   .319,   .297,   .276, .256,   .237,   .218,   .202,   .187, .172,   .158,   .145,   .132,   .122, .113,   .104,   .097,   .089,   .082, .077,   .072,   .065,   .059,   .057, .052,   .049,   .046,   .042,   .039, .037,   .034,   .032,   .029,   .027, .026,   .024,   .023,   .021,   .019, .018,   .017,   .017,   .016,   .016, .015,   .014,   .013,   .012,   .011, .010,   .010,   .009,   .009,   .008, .008,   .007,   .007,   .006,   .006, .005,   .005,   .005,   .004,   .004, .004,   .004,   .003,   .003,   .003, .003,   .003,   .002,   .002,   .002])
    psfrads = np.linspace(0,psf.size-1,num=psf.size)

    #arbitrary scaling to get better numerical integration
    width = width * 20
    height = height * 20
    xoff = 40*xcenter
    yoff = 40*ycenter

    yvs, xvs = np.mgrid[-99:100,-99:100]
    
    xvs -= xoff
    yvs -= yoff

    rs = xvs**2 + yvs**2
    rs = np.sqrt(rs)
    
    dx = np.abs(xvs)
    dy = np.abs(yvs)

    inside = np.where((dy < height) & (dx < width))
    outside = np.where((dy > height) | (dx > width))
    edgey = np.where((dy == height) & (dx <= width))
    edgex = np.where((dy <= height) & (dx == width))
    
    fluxes = np.zeros_like(rs)
    mod = interpolate.splrep(psfrads,psf,s=0)
    fluxes[np.where(rs<=99)] = interpolate.splev(rs[np.where(rs<=99)],mod,der=0)

    insideflux = fluxes[inside].sum()
    outsideflux = fluxes[outside].sum()
    edgexflux = fluxes[edgex].sum()
    edgeyflux = fluxes[edgey].sum()

    insideflux += 0.5*(edgexflux + edgeyflux)
    outsideflux += 0.5*(edgexflux + edgeyflux)
    
    fraction = insideflux / (insideflux + outsideflux)
    
    return fraction

def Extract1Dspec(filename,center_y,ywidth,numsig=4,showplots=False, slitwidth=0.7, slitlength=46, pixscale=0.1798, slcor=True,error=False):
    #note that slcor is a flag that determines whether to correct for slits losses
    #slitwidth and slitlength are in units of arcseconds
    
    # read in the data from the final 2D spec
    hdulist = fits.open(filename)
    scidata = hdulist[0].data
    
    ywidth = int(ywidth)
    if ywidth%2!=0:
        ywidth = ywidth -1
    
    #restrict data to prescribed yrange (only the rows around the trace)
    # collapse along the x direction so that all you have left is an average or median flux in the spatial direction)
    meddat = np.nanmedian(scidata,axis=1)[center_y-(ywidth/2):center_y+(ywidth/2)]

    #match up meddat and meandat arrays to physical pixel values
    pixelnum = np.arange(len(meddat))+center_y-(ywidth/2)
    #save the number of counts in each row (of the collapsed spectrum)
    countval1 = meddat

    #fit the shape of the flux distribution (in the spatial direction)
    #this will return the three parameters a,b,c that are used in the equation of a gaussian
    # you need to give it an initial guess p0 (a is the normalization, b is the centroid and c is the standard deviation)
    popt1, pcov1 = curve_fit(gaussian, pixelnum, countval1,p0=[20,center_y,4])
    print popt1


    #plot the fitted gaussian for the two different models and the data
    pix_fit = np.linspace(pixelnum[0], pixelnum[-1], 100)
    count_fit1 = gaussian(pix_fit, *popt1)

    if showplots:
        plt.plot(pixelnum,countval1,'sb')
        plt.plot(pix_fit, count_fit1,'r-o')
        plt.show()
    
    # now extract 1D spectrum
    # sum up all the flux between 4 sigma on either side of the spectrum:
    
    trace_med = scidata[popt1[1]-numsig*np.abs(popt1[2]):popt1[1]+numsig*np.abs(popt1[2]),:]
    spec1D_med = np.sum(trace_med,axis=0) 
    
    
    
    if slcor:
        c = popt1[2]
        fwhm_pix = np.abs(2*c)
        fwhm_arcsec = fwhm_pix*pixscale
        fracloss = GausSlitLoss(slitwidth/fwhm_arcsec,slitlength/fwhm_arcsec)
        spec1D_corr = spec1D_med/fracloss

    #calculate wavelength coverage
    startlambda = hdulist[0].header['CRVAL1']
    pixscale = hdulist[0].header['CD1_1']
    endlambda = startlambda+(len(spec1D_med)*pixscale)
    wavelength = np.linspace(startlambda,endlambda,len(spec1D_med))

    if showplots:
        if slcor:
            plt.plot(wavelength,spec1D_corr,'-b')
            #plt.plot(wavelength,spec1D_mean,'-k')
            plt.xlabel(r"Wavelength (in $\AA$)")
            plt.ylabel("e-/s")
            plt.title("Spectrum (corrected for slitlosses)")
        
        else:
            #print wavelength.shape
            plt.plot(wavelength,spec1D_med,'-b')
            #plt.plot(wavelength,spec1D_mean,'-k')
            plt.xlabel(r"Wavelength (in $\AA$)")
            plt.ylabel("e-/s")
            plt.title("Spectrum (not corrected for slitlosses)")
        
        plt.show()
    
    if slcor:
        return wavelength,spec1D_corr
    else:
        return wavelength,spec1D_med

def CalcFluxConvKband(template,standard,center_y,ywidth,numsig=4,temp_flux=0,stand_mag=5.484,zp_vega=3.961e-11,zp_2mass=4.283e-11,rangemin=19000,rangemax=25000):
    #load in the Vega template
    # my Vega template
    hdulist = fits.open(template)
    scidata = hdulist[1].data
    #print scidata.columns # note that the unit FLAM is ergs/cm^2/s/angs
    #indices where the wavelength is between 19000 and 25000 ANGs
    wind= (scidata['wavelength'] > rangemin) & (scidata['wavelength'] <rangemax)
    wave = scidata['wavelength'][wind]
    f_vega = np.array(scidata['flux'][wind])
    m_vega = 2.5*np.log10(zp_2mass/zp_vega)
    m_hip = stand_mag #in 2mass Ks band

    norm = np.power(10,-0.4*(m_hip-m_vega))

    f_HIP = f_vega*norm
    
    wavestand,spec1D_stand = Extract1Dspec(standard,center_y=center_y,ywidth=ywidth,numsig=4)
    #resample template:
    func_templ = interp1d(wave,f_HIP)
    #calculate the template flux values at the data points
    f_templ = func_templ(wavestand)
    conv_array = f_templ/spec1D_stand
    return wavestand,conv_array
    
def CalcMagKband(wavestar,spec1D_star,wavestand,conv_array,wavevega,flux_vega,respfile=os.path.join(os.path.expanduser('~'),'Dropbox','camille_x','Desktop','2mass_K.dat')):
    #calculate the K band magnitude of a point source
    waveresp, resp =  np.genfromtxt(respfile, unpack='True')
    
    #import vega spectrum:
    
    #function that can calculate the conversion array for arbitrary wavelength:
    conv_func = interp1d(wavestand,conv_array)
    #convert stellar spectrum from e-/s units to Flam
    flux_star_temp =  spec1D_star*conv_func(wavestar)

    qind = np.logical_not((np.isnan(flux_star_temp)))
    wavelength_star = wavestar[qind]
    flux_star=flux_star_temp[qind]
    
    #resample vega spectrum to stellar wavelengths
    vega_func = interp1d(wavevega,flux_vega)
    flux_vega_resamp = vega_func(wavelength_star)
    
    #resample the response function
    response_func = interp1d(waveresp,resp,fill_value=0, bounds_error=False)
    resp_resamp = response_func(wavelength_star)
    num_star = np.trapz(resp_resamp*flux_star,x=wavelength_star)
    denom_star = np.trapz(resp_resamp*flux_vega_resamp,x=wavelength_star)
    
    ratio_star = num_star/denom_star
    
    mag_star = -2.5*np.log10(ratio_star)
    return mag_star

def CalcBoxFlux(hdulist,lambda_min,lambda_max,center_spatial,width_spat,wavestand,conv_array,pstamp=False):
    
    width_spat = int(width_spat)
    
    if width_spat%2!=0:
        width_spat = width_spat-1
    
    scidata = hdulist[0].data
    box =scidata[center_spatial-(width_spat/2):center_spatial+(width_spat/2),lambda_min:lambda_max]
    if pstamp:
        with sns.axes_style("white"):
            plt.imshow(box,cmap='viridis',origin='lowerleft')
            plt.colorbar()
            plt.show()
    
    startlambda = hdulist[0].header['CRVAL1']
    pixscale = hdulist[0].header['CD1_1']
    endlambda = startlambda+(scidata.shape[1]*pixscale)
    wavelength = np.linspace(startlambda,endlambda,scidata.shape[1])

    wave = wavelength[lambda_min:lambda_max]
    eps_lam = np.zeros(len(wave))
    eps_lam = np.sum(box,axis=0)
    
    conv_func = interp1d(wavestand,conv_array)
    #convert stellar spectrum from e-/s units to Flam
    flux_lam =  eps_lam*conv_func(wavelength[lambda_min:lambda_max])

    #in FLAM units
    total_flux = np.sum(flux_lam)
    return total_flux

def CalcBoxError(hdulist,lambda_min, lambda_max,center_spatial,width_spat,wavestand,conv_array,renorm = False, renormfactor = 1):
    
    width_spat = int(width_spat)
    if width_spat%2!=0:
        width_spat = width_spat-1
    
    #note that the units of the 'Slug_K_SLUG_sig.fits' file is in units of e-/s
    stddata = hdulist[0].data
    if renorm:
        stddata=stddata*renormfactor
    
    stdbox = stddata[center_spatial-(width_spat/2):center_spatial+(width_spat/2),lambda_min:lambda_max]
    #convert from (e-/s) to flux
        
    startlambda = hdulist[0].header['CRVAL1']
    pixscale = hdulist[0].header['CD1_1']
    endlambda = startlambda+(stddata.shape[1]*pixscale)
    wavelength = np.linspace(startlambda,endlambda,stddata.shape[1])
    
    conv_func = interp1d(wavestand,conv_array)
    #convert stellar spectrum from e-/s units to Flam
    std_flux_box =  stdbox*conv_func(wavelength[lambda_min:lambda_max])

    #convert to variance so that the noise can be added:
    var_flux_box = std_flux_box**2

    #sum the variances (that are in flux^2 units)
    var_tot = np.sum(var_flux_box)

    #take square root of value: 
    std_tot = np.sqrt(var_tot)
    return std_tot

def CalcEmpiricalError(hdulist,lambda_min, lambda_max,center_spatial,width_spat,wavestand,conv_array):
    
    width_spat = int(width_spat)
    if width_spat%2!=0:
        width_spat = width_spat-1
    
    #filename = os.path.join(os.path.expanduser('~'),'Dropbox','camille_x','Desktop','Slug_K_SLUG_sig.fits')
    # read in the data from the final 2D spec
    data = hdulist[0].data

    skybox = data[center_spatial-(width_spat/2):center_spatial+(width_spat/2),lambda_min:lambda_max]
    #convert from (e-/s) to flux
        
    startlambda = hdulist[0].header['CRVAL1']
    pixscale = hdulist[0].header['CD1_1']
    endlambda = startlambda+(data.shape[1]*pixscale)
    wavelength = np.linspace(startlambda,endlambda,data.shape[1])
    
    conv_func = interp1d(wavestand,conv_array)
    #convert stellar spectrum from e-/s units to Flam
    sky_flux_box =  skybox*conv_func(wavelength[lambda_min:lambda_max])
    std_sky = np.std(sky_flux_box)

    std_sky_tot = std_sky*np.sqrt(width_spat*(lambda_max-lambda_min))
    return std_sky_tot

def CalcFluxSpatial(pixstart,pixend,hdulist1,hdulist2,lmin,lmax,wavestand,conv_array,width_spat=10,slit_width=1,pixscale=0.1798,centerwl=1075,pstamp=False):
    
    width_spat = int(width_spat)
    if width_spat%2!=0:
        width_spat = width_spat-1
    
    spatial_width = pixscale*width_spat
    
    numstep = ((pixend-pixstart)/width_spat)+1
    center_spat_array = np.linspace(pixstart,pixend,numstep)
    
    print center_spat_array
    fluxarray = np.zeros(len(center_spat_array))
    SBarray =  np.zeros(len(center_spat_array))
    errorfluxarray =  np.zeros(len(center_spat_array))
    errorSBarray =  np.zeros(len(center_spat_array))
    errorempirarray =  np.zeros(len(center_spat_array))
    errorempirSBarray =  np.zeros(len(center_spat_array))
    
    for j in xrange(len(center_spat_array)):
        center_spat = center_spat_array[j] 

        flux_tot = CalcBoxFlux(hdulist=hdulist1,lambda_min=lmin, lambda_max =lmax,center_spatial=center_spat,width_spat=width_spat,wavestand=wavestand,conv_array=conv_array,pstamp=pstamp)
        ferror_tot = CalcBoxError(hdulist=hdulist2,lambda_min=lmin, lambda_max = lmax,center_spatial=center_spat,width_spat=width_spat,wavestand=wavestand,conv_array=conv_array)
        err_empir_tot = CalcEmpiricalError(hdulist=hdulist1,lambda_min=lmin, lambda_max = lmax,center_spatial=center_spat,width_spat=width_spat,wavestand=wavestand,conv_array=conv_array)

        fluxarray[j] = flux_tot    
        SBarray[j] = flux_tot/(spatial_width*slit_width)

        errorfluxarray[j] = ferror_tot    
        errorSBarray[j] = ferror_tot/(spatial_width*slit_width)

        errorempirarray[j] = err_empir_tot
        errorempirSBarray[j] = err_empir_tot/(spatial_width*slit_width)
    return fluxarray,SBarray,errorfluxarray,errorSBarray,errorempirarray,errorempirSBarray,center_spat_array
    
def CalcFluxSpatialCorr(pixstart,pixend,spacing,hdulist1,hdulist2,lmin,lmax,wavestand,conv_array,width_spat=10,slit_width=1,pixscale=0.1798,centerwl=1075,pstamp=False):
    #Like CalcFluxSpatial but with flexible spacing between measurements (you can make the boxes correlated)
    width_spat = int(width_spat)
    if width_spat%2!=0:
        width_spat = width_spat-1
    
    spatial_width = pixscale*width_spat
    
    numstep = ((pixend-pixstart)/spacing)+1
    center_spat_array = np.linspace(pixstart,pixend,numstep)
    
    print center_spat_array
    fluxarray = np.zeros(len(center_spat_array))
    SBarray =  np.zeros(len(center_spat_array))
    errorfluxarray =  np.zeros(len(center_spat_array))
    errorSBarray =  np.zeros(len(center_spat_array))
    errorempirarray =  np.zeros(len(center_spat_array))
    errorempirSBarray =  np.zeros(len(center_spat_array))
    
    for j in xrange(len(center_spat_array)):
        center_spat = center_spat_array[j] 

        flux_tot = CalcBoxFlux(hdulist=hdulist1,lambda_min=lmin, lambda_max =lmax,center_spatial=center_spat,width_spat=width_spat,wavestand=wavestand,conv_array=conv_array,pstamp=pstamp)
        ferror_tot = CalcBoxError(hdulist=hdulist2,lambda_min=lmin, lambda_max = lmax,center_spatial=center_spat,width_spat=width_spat,wavestand=wavestand,conv_array=conv_array)
        err_empir_tot = CalcEmpiricalError(hdulist=hdulist1,lambda_min=lmin, lambda_max = lmax,center_spatial=center_spat,width_spat=width_spat,wavestand=wavestand,conv_array=conv_array)

        fluxarray[j] = flux_tot    
        SBarray[j] = flux_tot/(spatial_width*slit_width)

        errorfluxarray[j] = ferror_tot    
        errorSBarray[j] = ferror_tot/(spatial_width*slit_width)

        errorempirarray[j] = err_empir_tot
        errorempirSBarray[j] = err_empir_tot/(spatial_width*slit_width)
    return fluxarray,SBarray,errorfluxarray,errorSBarray,errorempirarray,errorempirSBarray,center_spat_array
    
def CalcBoxFluxFA(fluxedarray,lambda_min,lambda_max,spat_min,spat_max,pstamp=False):
    #same as CalcBoxFlux but with fluxed array input    
    box =fluxedarray[spat_min:spat_max,lambda_min:lambda_max]
    if pstamp:
        with sns.axes_style("white"):
            plt.imshow(box,cmap='viridis',origin='lowerleft')
            plt.colorbar()
            plt.show()

    #in FLAM units
    total_flux = np.sum(box)
    return total_flux

def CalcBoxErrorFA(stddata,lambda_min,lambda_max,spat_min,spat_max):
    
    stdbox = stddata[spat_min:spat_max,lambda_min:lambda_max]
    
    #convert to variance so that the noise can be added:
    varbox = stdbox**2

    #sum the variances (that are in flux^2 units)
    var_tot = np.sum(varbox)

    #take square root of value: 
    std_tot = np.sqrt(var_tot)
    return std_tot

def CalcEmpiricalErrorFA(fluxedarray,lambda_min,lambda_max,spat_min,spat_max):
    
    skybox = fluxedarray[spat_min:spat_max,lambda_min:lambda_max]
    std_sky = np.std(skybox)

    std_sky_tot = std_sky*np.sqrt((spat_max-spat_min)*(lambda_max-lambda_min))
    return std_sky_tot

def CalcFluxSpatialCorrFA(pixstart,pixend,spacing,fluxedarray,stddata,lmin,lmax,width_spat=10,slit_width=1,pixscale=0.1798,pstamp=False):
    #Like CalcFluxSpatial but with flexible spacing between measurements (you can make the boxes correlated) and you can give the fluxed spectrum as an array (skips the hdulist steps and the conversion array that fluxes the spectrum)
    width_spat = int(width_spat)
    if width_spat%2!=0:
        width_spat = width_spat-1

    spatial_width = pixscale*width_spat

    numstep = ((pixend-pixstart)/spacing)+1
    center_spat_array = np.linspace(pixstart,pixend,numstep)

    print center_spat_array
    fluxarray = np.zeros(len(center_spat_array))
    SBarray =  np.zeros(len(center_spat_array))
    errorfluxarray =  np.zeros(len(center_spat_array))
    errorSBarray =  np.zeros(len(center_spat_array))
    errorempirarray =  np.zeros(len(center_spat_array))
    errorempirSBarray =  np.zeros(len(center_spat_array))

    for j in xrange(len(center_spat_array)):
        center_spat = center_spat_array[j]
        smin = center_spat-(width_spat/2)
        smax = center_spat+(width_spat/2)
        
        flux_tot = CalcBoxFluxFA(fluxedarray=fluxedarray,lambda_min=lmin,lambda_max=lmax,spat_min=smin,spat_max=smax,pstamp=pstamp)
        ferror_tot = CalcBoxErrorFA(stddata=stddata,lambda_min=lmin,lambda_max=lmax,spat_min=smin,spat_max=smax)
        
        err_empir_tot = CalcEmpiricalErrorFA(fluxedarray=fluxedarray,lambda_min=lmin,lambda_max=lmax,spat_min=smin,spat_max=smax)

        fluxarray[j] = flux_tot
        SBarray[j] = flux_tot/(spatial_width*slit_width)

        errorfluxarray[j] = ferror_tot
        errorSBarray[j] = ferror_tot/(spatial_width*slit_width)

        errorempirarray[j] = err_empir_tot
        errorempirSBarray[j] = err_empir_tot/(spatial_width*slit_width)
        
    return fluxarray,SBarray,errorfluxarray,errorSBarray,errorempirarray,errorempirSBarray,center_spat_array

#scidata is an np.array with the data

def Save_FITS_Image_WCS_Cutout(scidata,header, Xbounds, Ybounds, outfile='tst_fig.pdf'):

    ''' Image '''
    # Cut
    xmin = Xbounds[0]
    xmax = Xbounds[1]
    ymin = Ybounds[0]
    ymax = Ybounds[1]
    cut_img = scidata[ymin:ymax,xmin:xmax]   

    #calculate new WCS zeropoint:
    #save old CRPIX1 and CRPIX2
    oldcrpix1 = header['CRPIX1']
    oldcrpix2 = header['CRPIX2']
    
    newcrpix1 = oldcrpix1 - xmin
    newcrpix2 = oldcrpix2 - ymin
    
    header['CRPIX1'] = newcrpix1
    header['CRPIX2'] = newcrpix2
    
    hdu = fits.PrimaryHDU()
    hdu.data = cut_img
    hdu.header = header
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(outfile)
    hdulist.close()
    
def FindCentroid(filename,center_y,ywidth,numsig=4,showplots=True, slitwidth=0.7, slitlength=46, pixscale=0.1798):
    # read in the data from the final 2D spec
    hdulist = fits.open(filename)
    scidata = hdulist[0].data
    
    #check that ywidth is even, if not, subtract 1:
    ywidth = int(ywidth)
    if ywidth%2!=0:
        ywidth = ywidth -1
    
    #restrict data to prescribed yrange (only the rows around the trace)
    # collapse along the x direction so that all you have left is an average or median flux in the spatial direction)
    meddat = np.nanmedian(scidata,axis=1)[center_y-(ywidth/2):center_y+(ywidth/2)]

    #match up meddat and meandat arrays to physical pixel values
    pixelnum = np.arange(len(meddat))+center_y-(ywidth/2)
    #save the number of counts in each row (of the collapsed spectrum)
    countval1 = meddat

    #fit the shape of the flux distribution (in the spatial direction)
    #this will return the three parameters a,b,c that are used in the equation of a gaussian
    # you need to give it an initial guess p0 (a is the normalization, b is the centroid and c is the standard deviation)
    popt1, pcov1 = curve_fit(gaussian, pixelnum, countval1,p0=[20,center_y,numsig])
    if showplots:
        #plot the fitted gaussian for the two different models and the data
        pix_fit = np.linspace(pixelnum[0], pixelnum[-1], 100)
        count_fit1 = gaussian(pix_fit, *popt1)
        
        plt.plot(pixelnum,countval1,'sb')
        plt.plot(pix_fit, count_fit1,'r-o')
        plt.show()
    print popt1
    return popt1
    
def FindCentroidX(filename,center_x,xwidth,showplots=True, pixscale=0.135):
    # read in the data from the final 2D spec
    hdulist = fits.open(filename)
    scidata = hdulist[0].data
    
    #check that xwidth is even, if not, subtract 1:
    xwidth = int(xwidth)
    if xwidth%2!=0:
        xwidth = xwidth -1
    
    #restrict data to prescribed yrange (only the rows around the trace)
    # collapse along the y direction so that all you have left is an average or median flux in the spatial direction)
    meddat = np.nanmedian(scidata,axis=0)[center_x-(xwidth/2):center_x+(xwidth/2)]

    #match up meddat and meandat arrays to physical pixel values
    pixelnum = np.arange(len(meddat))+center_x-(xwidth/2)
    #save the number of counts in each row (of the collapsed spectrum)
    countval1 = meddat

    #fit the shape of the flux distribution (in the spatial direction)
    #this will return the three parameters a,b,c that are used in the equation of a gaussian
    # you need to give it an initial guess p0 (a is the normalization, b is the centroid and c is the standard deviation)
    popt1, pcov1 = curve_fit(gaussian, pixelnum, countval1,p0=[20,center_x,8])
    if showplots:
        #plot the fitted gaussian for the two different models and the data
        pix_fit = np.linspace(pixelnum[0], pixelnum[-1], 100)
        count_fit1 = gaussian(pix_fit, *popt1)
        
        plt.plot(pixelnum,countval1,'sb')
        plt.plot(pix_fit, count_fit1,'r-o')
        plt.show()
    print popt1
    return popt1

def ShiftDataX(shift,dataref,datashift):
    #make matrix of zeros that is same size as ref
    shifteddata = np.zeros_like(dataref)
    shift = int(shift)
    if shift > 0:
        #shifteddata[0:(-1*shift),:] = datashift[shift:,:]
        shifteddata[:,0:int(-1*shift)] = datashift[:,shift:]
    
    elif shift < 0:
        shifteddata[:,int(-1*shift):] = datashift[:,:shift]
    
    else:
        shifteddata[:,:] = datashift
    return shifteddata


def ShiftData(shift,dataref,datashift):
    #make matrix of zeros that is same size as ref
    shifteddata = np.zeros_like(dataref)
    shift = int(shift)
    if shift > 0:
        shifteddata[0:(-1*shift),:] = datashift[shift:,:]
    
    elif shift < 0:
        shifteddata[(-1*shift):,:] = datashift[:shift,:]
    
    else:
        shifteddata[:,:] = datashift
    return shifteddata