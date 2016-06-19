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
    tampl, tcent, twid, w, yprep = find_peaks(spec)   # OLD CODE, NO PROB
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
    pixwave = cheby_val(s['calib'][idx]['ffit'], pixpk,   # OLD CODE, NO PROB
                        s['calib'][idx]['nrm'],s['calib'][idx]['nord'])
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

