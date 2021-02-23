import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer as kbd

infile = sys.argv[1]
outfile = sys.argv[2]

lund_jets = np.load( infile, allow_pickle=True )

def get_plane( lund_jet_sample, plane_id ):
    return [ [ [i[6],i[7]] for i in jet if i[0]==plane_id ] for jet in lund_jet_sample ]

lund_0 = get_plane( lund_jets, 0)
lund_1 = get_plane( lund_jets, 1)
lund_2 = get_plane( lund_jets, 2)
lund_3 = get_plane( lund_jets, 3)

planes = 3

if planes == 0:
    lund_d = lund_0
if planes == 1:
    lund_d = [ i+j for i,j in zip(lund_0,lund_1)  ]
if planes == 2:
    lund_d = [ i+j+k for i,j,k in zip(lund_0,lund_1,lund_2)  ]
if planes == 3:
    lund_d = [ i+j+k+l for i,j,k,l in zip(lund_0,lund_1,lund_2,lund_3) ]

masses = np.array( [ np.max( [ i[2] for i in jet ] ) for jet in lund_jets ] )

def flatten( lund_jet_sample ):
    jf = []
    for jet in lund_jet_sample:
        jf += jet
    return jf

lund_d_flat = flatten( lund_d )

def cut_to_range( data, cuts ):
    if len(data[0][0]) != len(cuts):
        print("ERROR: dimensions of cuts should match number of observables")
        sys.exit()
    new_data = []
    for jet in data:
        new_jet = []
        for splitting in jet:
            keep = True
            for i in range(len(splitting)):
                if cuts[i]==None:
                    pass
                elif splitting[i]<cuts[i][0] or splitting[i]>cuts[i][1]:
                    keep = False
            if keep == True:
                new_jet.append(splitting)
        if len(new_jet)>0:
            new_data.append(new_jet)
    return new_data

lund_d_cut = cut_to_range( lund_d, ((0,7),(-2,5)) )
lund_d_flat_cut = flatten( lund_d_cut )

transform = kbd( n_bins=[40,40], encode='ordinal', strategy='uniform' )
transform.fit( lund_d_flat_cut )

lund_d_ord = [ transform.transform(i) for i in lund_d_cut ]
lund_d_flat_ord = transform.transform( lund_d_flat_cut )

def ord2onehot( data_ord, lims ):
    img = np.zeros((lims[0],lims[1]))
    for i in data_ord:
        img[ lims[1] - 1 - int(i[1]), int(i[0]) ] = 1
    return img

lund_d_onehot = []
for jet in lund_d_ord:
    lund_d_onehot.append( ord2onehot( jet, [40,40] ).reshape(-1) )
lund_d_onehot = np.array( lund_d_onehot )

lund_d_onehot_2 = np.array( [ np.concatenate( ( lund_d_onehot[i], np.array([0]), [masses[i]] ), axis=-1 )
                                 for i in range(len(lund_d_onehot)) ] )

out = pd.DataFrame( lund_d_onehot_2, columns=[str(i) for i in range(1600)]+["__signal_col__"]+["__mass_Col__"] )

out.to_hdf( outfile+".h5", "table", format="table", complib="blosc", complevel=5 )

