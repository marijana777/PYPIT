# Module to run tests on ararclines

# import
import json, os
import numpy as np


from pypit import pyputils
msgs = pyputils.get_dummy_logger()

from pypit import arholy as holy
from pypit import arparse as settings


from pypit import arutils as arut
arut.dummy_settings()
slf = arut.dummy_self()

# LRIS R600

lamps = ['ArI','NeI','HgI','KrI','XeI']
disperser = '600/7500'

# Archived solution
soln_file = os.getenv('DROPBOX_DIR')+'/PYPIT/Holy_Grail/lrisr_600_7500_holy.json'
with open(soln_file,'r') as f:
    soln = json.load(f)

holy.holy1(np.array(soln['spec']), lamps, disperser)