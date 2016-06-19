# To get this running, you must do the following at the command line:
# python arcytrace_setup.py build_ext --inplace
# although I'm not really sure what the --inplace does, I think it means "only valid for this directory"

import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.float64
ctypedef np.float_t DTYPE_t
ITYPE = np.int64
ctypedef np.int_t ITYPE_t

cdef extern from "math.h":
	double csqrt "sqrt" (double)
	double cexp "exp" (double)
	double clog "log" (double)
	double cpow "pow" (double, double)

@cython.boundscheck(False)
def brute_force_solve(np.ndarray[DTYPE_t, ndim=1] pix not None,
						np.ndarray[DTYPE_t, ndim=1] linelist not None,
						np.ndarray[DTYPE_t, ndim=1] coeff not None,
						double npix, double lim):

	cdef int ii, jj, pp, npx, qq, nl
	cdef int bii, bjj, bidx
	cdef double bval, tval, bscr, tscr

	npx = pix.shape[0]
	nl = linelist.shape[0]

	# Define the range of coefficients available
	cdef int nsz = 2*<int>(lim*npix) # Brute force to within 0.5 pixels
	# Note that lim = 1.0 would be very conservative, and would encompass
	# all possible values.
	cdef np.ndarray[DTYPE_t, ndim=1] cval = np.linspace(-lim, lim, nsz)
	cdef np.ndarray[DTYPE_t, ndim=1] llp = np.zeros((nl), dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] ll = np.zeros((nl), dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] lll = np.zeros((nl), dtype=DTYPE)
	for qq in range(nl):
		ll[qq] = linelist[qq]*linelist[qq]
		lll[qq] = linelist[qq]*linelist[qq]*linelist[qq]

	# Create a list of the IDs
	#cdef np.ndarray[DTYPE_t, ndim=1] diffids = np.zeros((npx), dtype=DTYPE)
	cdef np.ndarray[ITYPE_t, ndim=1] ids = np.zeros((npx), dtype=ITYPE)

	# Iterate through the brute force calculation
	for ii in range(nsz):
		for jj in range(nsz):
			# Convert the linelist into pixels
			for qq in range(nl):
				llp[qq] = coeff[0] + linelist[qq]*coeff[1] + ll[qq]*cval[ii] + lll[qq]*cval[jj]
			# For each pixel, find the closest ID in the linelist
#			for pp in range(npx):
#				bval = pix[pp]-llp[0]
#				if bval < 0.0: bval *= -1.0
#				bidx = 0
			bidx = 0
			tscr = 0.0
			for pp in range(npx):
				bval = pix[pp]-llp[bidx]
				if bval < 0.0:
					bval *= -1.0
				if bidx+1 >= nl:
					break
				for qq in range(bidx+1, nl):
					tval = pix[pp]-llp[qq]
					if tval < 0.0:
						tval *= -1.0
					if tval < bval:
						bval = tval
						bidx = qq
					else:
						# Because ll is sorted, we can break the loop
						# here if a worse value isn't found
						break
				# Store the best residual of this pixel
				tscr += bval
			# Calculate the score of this solution and test
			# if it's better than the rest
			#tscr = np.median(diffids)
			if (ii==0 and jj==0) or (tscr < bscr):
				bscr = tscr
				bii = ii
				bjj = jj
	#print cval[bii], cval[bjj]
	# Once the best solution is found, ID the lines
	# Convert the linelist into pixels
	for qq in range(nl):
		llp[qq] = coeff[0] + linelist[qq]*coeff[1] + ll[qq]*cval[bii] + lll[qq]*cval[bjj]
	# For each pixel, find the closest ID in the linelist
	for pp in range(npx):
		bval = pix[pp]-llp[0]
		if bval < 0.0:
			bval *= -1.0
		bidx = 0
		for qq in range(1,nl):
			tval = pix[pp]-llp[qq]
			if tval < 0.0:
				tval *= -1.0
			if tval < bval:
				bval = tval
				bidx = qq
			else:
				# Because ll is sorted, we can break the loop
				# here if a better value isn't found
				break
		# Assign the best ID to this pixel
		ids[pp] = bidx
	# Return the best indices
	return ids
