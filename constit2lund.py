import os
import sys
import numpy as np
import pandas as pd
from preprocess_qcd_top_jets import *

h5path = sys.argv[1]
njets = int( eval( sys.argv[2] ) )
outfile = sys.argv[3]

R=0.8
p=-1
jets = cluster_toptagging_dataset_part( h5path, R, p, [0,njets] )

qcd_jets = jets['qcd']
top_jets = jets['top']

R=1.0
p=0
qcd_lund_jets = []
for jet in qcd_jets:
    lund_jet = get_lund_history( jet, R, p )
    lund_splittings = get_lund_splittings( lund_jet )
    qcd_lund_jets.append( lund_splittings )
top_lund_jets = []
for jet in top_jets:
    lund_jet = get_lund_history( jet, R, p )
    lund_splittings = get_lund_splittings( lund_jet )
    top_lund_jets.append( lund_splittings )

qcd_lund_jets = np.array( qcd_lund_jets, dtype=object )
top_lund_jets = np.array( top_lund_jets, dtype=object )

np.save( "qcd_"+outfile+".npy", qcd_lund_jets )
np.save( "top_"+outfile+".npy", top_lund_jets )
