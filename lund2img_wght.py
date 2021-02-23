import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer as kbd

infile = sys.argv[1]
outfile = sys.argv[2]
in_wght = eval(sys.argv[3])

lund_jets = np.load( infile, allow_pickle=True )

# the default weight=1 corresponds to pt, 2 is mass, 3 is mass drop, ...
def get_plane_w( lund_jet_sample, plane_id, weight=in_wght):
    return [ [ [i[6],i[7],i[weight]] for i in jet if i[0]==plane_id ] for jet in lund_jet_sample ]

lund_0 = get_plane_w( lund_jets, 0 )
lund_1 = get_plane_w( lund_jets, 1 )
lund_2 = get_plane_w( lund_jets, 2 )
lund_3 = get_plane_w( lund_jets, 3 )

masses = np.array( [ np.max( [ i[2] for i in jet ] ) if len(jet)>0 else 0.0 for jet in lund_jets ] )

def flatten( lund_jet_sample ):
    jf = []
    for jet in lund_jet_sample:
        jf += jet
    return jf

def flatten_w( lund_jet_sample ):
    jf = []
    for jet in lund_jet_sample:
        jf += [ i[0:2] for i in jet ]
    return jf

lund_0_flat = flatten_w( lund_0 )
lund_1_flat = flatten_w( lund_1 )
lund_2_flat = flatten_w( lund_2 )
lund_3_flat = flatten_w( lund_3 )

def cut_to_range_w( data, cuts, dim=2):
    if dim != len(cuts):
        print("ERROR: dimensions of cuts should match number of observables")
        sys.exit()
    new_data = []
    weights = []
    for jet in data:
        new_jet = []
        new_wghts = []
        for splitting in jet:
            keep = True
            for i in range(len(splitting)-1):                
                if cuts[i]==None:
                    pass
                elif splitting[i]<cuts[i][0] or splitting[i]>cuts[i][1]:
                    keep = False
            if keep == True:
                new_jet.append(splitting[0:2])
                new_wghts.append(splitting[2])
        if len(new_jet)>0:
            new_data.append(new_jet)
            weights.append(new_wghts)
    return new_data, weights

lund_0_cut, lund_0_wghts_cut = cut_to_range_w( lund_0, ((0,7),(-2,6)) )
lund_0_flat_cut = flatten( lund_0_cut )

transform = kbd( n_bins=[40,40], encode='ordinal', strategy='uniform' )
transform.fit( lund_0_flat_cut )

lund_0_ord = [ transform.transform(i) for i in lund_0_cut ]
lund_0_flat_ord = transform.transform( lund_0_flat_cut )

def ord2onehot( data_ord, lims ):
    img = np.zeros((lims[0],lims[1]))
    for i in data_ord:
        img[ lims[1] - 1 - int(i[1]), int(i[0]) ] = 1
    return img

def ord2onehot_w(data_ord, lims, weights, norm=True):
    img = np.zeros((lims[0],lims[1]))
    for i in range(len(data_ord)):
        img[ lims[1] - 1 - int( data_ord[i][1] ), int( data_ord[i][0] ) ] = weights[i]
    if norm == True:
        nn = np.sum(img)
        img = img/nn
    return img

lund_0_ohw = []
for jet,wght in zip(lund_0_ord,lund_0_wghts_cut):
    lund_0_ohw.append( ord2onehot_w( jet, [40,40], wght ).reshape(-1) )
lund_0_ohw = np.array( lund_0_ohw )

lund_0_ohw_2 = np.array( [ np.concatenate( ( lund_0_ohw[i], np.array([0]), [masses[i]] ), axis=-1 )
                                 for i in range(len(lund_0_ohw)) ] )

out = pd.DataFrame( lund_0_ohw_2, columns=[str(i) for i in range(1600)]+["__signal_col__"]+["__mass_Col__"] )

out.to_hdf( outfile+".h5", "table", format="table", complib="blosc", complevel=5 )

