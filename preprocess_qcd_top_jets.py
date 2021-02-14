import sys
import os
import numpy as np
import pyjet
import h5py     
from pyjet import cluster,DTYPE_PTEPM
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer as kbd

##########################
# function for clustering the top tagging dataset in chunks
##########################

def cluster_toptagging_dataset_part( h5file, R, p, evrange ):
    data_table = pd.read_hdf( h5file, start=evrange[0], stop=evrange[1], key="table" )
    alljets = {}
    alljets['top'] = []
    alljets['qcd'] = []
    for jet in data_table.iterrows():
        jet = jet[1]
        pseudojets_input = np.zeros( len( [0 for i in jet[::4] if i!=0] ), dtype=DTYPE_PTEPM )
        for j in range(len( pseudojets_input )):
            E = jet[j*4+0]
            px = jet[j*4+1]
            py = jet[j*4+2]
            pz = jet[j*4+3]
            pt = np.sqrt( px**2 + py**2 )
            if py != 0:
                phi = np.arctan( px/py )
            if not py > 0:
                phi = np.arctan( px/0.00001)
            p = np.sqrt( px**2 + py**2 + pz**2 )
            if p !=0:
                theta = np.arccos( pz/p )
            if not p>0:
                theta = np.arccos( pz/0.00001 )
            eta = np.log( np.arctan( theta/2 ) )
            #mass = np.sqrt( E**2 - p**2 )
            pseudojets_input[j]['pT'] = pt
            pseudojets_input[j]['mass'] = 0.0
            pseudojets_input[j]['eta'] = eta
            pseudojets_input[j]['phi'] = phi
        sequence = cluster( pseudojets_input, R=R, p=p )
        jets = sequence.inclusive_jets( ptmin=20 )
        final_jet = sorted( jets, key=lambda PseudoJet: PseudoJet.pt, reverse=True )[0]
        if jet['is_signal_new'] == True:
            alljets['top'] += [final_jet]
        else:
            alljets['qcd'] += [final_jet]
    return alljets

##########################
# function for getting the Lund history for a single jet, i.e. the clustering history but with each 4-momentum having a 'plane-id'
##########################

def get_lund_history( jet, R, p ):

    # re-clustering the jet
    clustered_jet = cluster( jet.constituents_array(), R=R, p=p )
    splittings=[]
    
    # each level in the clustering history is represented by a list of subjets
    # each subjet has [4-momentum,plane_id]
    lund_history = []

    # looping over the levels
    for t in range(0,clustered_jet.n_exclusive_jets(0)):

        # list momenta of all subjets at current splitting
        j_obs = [ [j.mass,j.pt,j.eta,j.phi] for j in clustered_jet.exclusive_jets(t) ]
        j_obs.sort( key = lambda i: i[1], reverse=True )
        j_obs_id = [ [j_obs[i],i] for i in range(len(j_obs)) ]

        if len(j_obs) <= 2:
            lund_history.append( j_obs_id )

        if len(j_obs)>2:
            # list momenta,id of all subjets at previous splitting
            pj_obs_id = [ j for j in lund_history[-1] ]
            pj_obs = [ j[0] for j in lund_history[-1] ]
            
            # work out which subjet split, and label it p_obs
            p_obs = [ pj_obs[i] for i in range(len(pj_obs)) if pj_obs[i] not in j_obs ]
            p_obs_id = [ pj_obs_id[i] for i in range(len(pj_obs)) if pj_obs[i] not in j_obs ]
            
            # work out which subjets didn't split, label them np_obs
            np_obs = [ pj_obs[i] for i in range(len(pj_obs)) if pj_obs[i] in j_obs ]
            np_obs_id = [ pj_obs_id[i] for i in range(len(pj_obs)) if pj_obs[i] in j_obs ]

            # work out what it split into, and put them in d_obs
            d_obs = [ j_obs[i] for i in range(len(j_obs)) if j_obs[i] not in pj_obs ]
            d_obs.sort( key = lambda i: i[1], reverse=True )
            pid = p_obs_id[0][1]
            d_obs_id = [ [d_obs[i],i+pid] for i in range(len(d_obs)) ]
            
            if len(d_obs) == 2 and len(p_obs) == 1:
                lund_history.append( np_obs_id + d_obs_id )

    return lund_history

##########################
# function which takes the lund history and computes the observables at all splittings in the jet, with a label indicating which plane it comes from
##########################

def get_lund_splittings( jet_history ):
    
    lund_splittings = []

    for i in range( len(jet_history)-1 ):
        
        # get info from current (j) and previous (pj) subjets in history
        pj_obs_id = jet_history[i]
        j_obs_id = jet_history[i+1]
        pj_obs = [ j[0] for j in pj_obs_id ]
        j_obs = [ j[0] for j in j_obs_id ]

        # get the subjet that split
        p_obs_id = [ pj_obs_id[i] for i in range(len(pj_obs)) if pj_obs[i] not in j_obs ]
        p_obs = [ i[0] for i in p_obs_id ]

        # get what it split into
        d_obs_id = [ j_obs_id[i] for i in range(len(j_obs)) if j_obs[i] not in pj_obs ]
        d_obs = [ i[0] for i in d_obs_id ]

        # check that len(d_obs) is 2, and len(p_obs) is 1, in case there is a very soft splitting where it appears that d_obs has just one component due the numerical accuracy
        # calculate observables
        if len(d_obs) == 2 and len(p_obs) == 1:
            pt = p_obs[0][1]
            pmass = p_obs[0][0]
            max_d_mass = max( d_obs[0][0], d_obs[1][0] )
            min_d_mass = min( d_obs[0][0], d_obs[1][0] )
            d_pts = [ d_obs[0][1], d_obs[1][1] ]
            max_d_pt = max( d_pts )
            min_d_pt = min( d_pts )
            plane_id = p_obs_id[0][1]
            try:
                mass_drop = max_d_mass/pmass
            except ZeroDivisionError:
                mass_drop = 0
            try:
                d_mass_ratio = min_d_mass/max_d_mass
            except ZeroDivisionError:
                d_mass_ratio = 0
            dR = np.sqrt( ( d_obs[0][2] - d_obs[1][2])**2 + (d_obs[0][3] - d_obs[1][3] )**2 )
            logidR = np.log( 1/dR )
            logkt = np.log( min_d_pt*dR )
            z = min_d_pt/(min_d_pt+max_d_pt)
            kap = z*dR
            
            # assign to a splitting
            splitting = [ plane_id, pt, pmass, mass_drop, d_mass_ratio, dR, logidR, logkt, z, kap ]

            # append to lund splittings
            lund_splittings.append( splitting )

    return lund_splittings

def create_basic_lund_image( lund_obs, bins ):

    # bins should be a 2D array for bins in x and y direction
    # x could correspond to logidR, y could correspond to logkt, preprocess the lund_obs first

    kbd_lund = kbd( n_bins=bins, encode='onehot', strategy='uniform' )
    kbd_lund.fit( lund_obs )
    lund_image = kbd_lund.transform( lund_obs )
    return lund_image, kbd_lund


