from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

#include_gsl_dir = "/Users/rcooke/local/include/"
#lib_gsl_dir = "/Users/rcooke/local/lib/"
include_gsl_dir = "/Users/xavier/anaconda/include/"
lib_gsl_dir = "/Users/xavier/anaconda/lib/"

ext = Extension("test_arcy", ["test_arcy.pyx"],
	include_dirs=[numpy.get_include(),
				include_gsl_dir],
	library_dirs=[lib_gsl_dir],
	libraries=["gsl","gslcblas"]
)

setup(ext_modules=[ext],
		cmdclass = {'build_ext': build_ext})
