#!/usr/bin/env python

# script written by Christof Sauer, using code written for https://arxiv.org/abs/1808.08979v3

# Standard imports
import sys
import time
import pprint
import argparse

# Scientific imports
import numpy as np
import pandas as pd

# Silence some annoying deprecation warnings
import warnings
warnings.filterwarnings("ignore")


"""
  Some global settings
"""

# Input/output settings
__signal_col__  = "is_signal_new"
__mass_col__    = "mass"
__batch_size__ = 1000
__n_warning__ = 0.7
__time_start__ = time.time()
n_shift_phi, n_shift_eta = 0, 0

# Grid settings
xpixels = np.arange(-2.6, 2.6, 0.029)
ypixels = np.arange(-np.pi, np.pi, 0.035)


# Calculate pseudorapidity of pixel entries
def eta (pT, pz):

  small = 1e-10
  small_pT = (np.abs(pT) < small)
  small_pz = (np.abs(pz) < small)
  not_small = ~(small_pT | small_pz)

  theta = np.arctan(pT[not_small]/pz[not_small])
  theta[theta < 0] += np.pi

  etas = np.zeros_like(pT)
  etas[small_pz] = 0
  etas[small_pT] = 1e-10
  etas[not_small] = np.log(np.tan(theta/2))
  return etas
 

# Calculate phi (in range [-pi,pi]) of pixel entries
def phi (px, py):

  """
    phis are returned in rad., np.arctan(0,0)=0 -> zero constituents set to -np.pi
  """

  phis = np.arctan2(py,px)
  phis[phis < 0] += 2*np.pi
  phis[phis > 2*np.pi] -= 2*np.pi
  phis = phis - np.pi 
  return phis
 

# Put eta-phi entries on grid
def orig_image (etas, phis, es):

  """
    Gives the value on grid with minimal distance,
    eg. for xpixel = (0,1,2,3,..) eta=1.3 -> xpixel=1, eta=1.6 ->xpixel=2
  """

  z = np.zeros((etas.shape[0],len(xpixels),len(ypixels)))
  in_grid = ~((etas < xpixels[0]) | (etas > xpixels[-1]) | (phis < ypixels[0]) | (phis > ypixels[-1]))
  xcoords = np.argmin(np.abs(etas[:,None,:] - xpixels[None,:,None]),axis=1)
  ycoords = np.argmin(np.abs(phis[:,None,:] - ypixels[None,:,None]),axis=1)
  ncoords = np.repeat(np.arange(etas.shape[0])[:,None],etas.shape[1],axis=1)
  z[ncoords[in_grid],ycoords[in_grid],xcoords[in_grid]] = es[in_grid]
  return z
 

def print_time (msg):

  print("[%8.2f] %s"%(time.time()-__time_start__,msg))


def img_mom (x, y, weights, x_power, y_power):

  """
    returns image momenta for centroid and principal axis
  """
  return ((x**x_power)*(y**y_power)*weights).sum()


def preprocessing (x ,y, weights, rotate=True, flip=True):

  """
    (x,y) are the coordinates and weights the corresponding values, shifts
    centroid to origin, rotates image, so that principal axis is vertical,
    flips image, so that most weights lay in (x<0, y>0)-plane.
    Method for calculating principal axis (similar to tensor of inertia):
    https://en.wikipedia.org/wiki/Image_moment
    here: y=phi, phi has modulo 2*np.pi but it's not been taken care of hear,
    so possible issues with calculating the centroid
    -> pre-shifting of events outside of this function solves the problem
    for iamge-data with Delta_phi < 2*np.pi
  """

  # Shift
  x_centroid = img_mom(x, y, weights, 1, 0) / weights.sum()
  y_centroid = img_mom(x, y, weights, 0, 1)/ weights.sum()
  x = x - x_centroid
  y = y - y_centroid

  # Check if shifting worked, there can be problems with modulo variables like phi (y)
  # x and y are sorted after highest weight, 0-comp. gives hottest event
  # for Jet-like Images Centroid should be close to hottest constituen (pT-sorted arrays)  
  global n_shift_phi
  global n_shift_eta
  if np.abs(x[0]) > __n_warning__:
    n_shift_eta += 1  
  if np.abs(y[0]) > __n_warning__:
    n_shift_phi += 1       

  if rotate:
    #Ccovariant matrix, eigenvectors corr. to principal axis
    u11 = img_mom(x, y, weights, 1, 1) / weights.sum()
    u20 = img_mom(x, y, weights, 2, 0) / weights.sum()
    u02 = img_mom(x, y, weights, 0, 2) / weights.sum()
    cov = np.array([[u20, u11], [u11, u02]])

    # Eigenvalues and eigenvectors of covariant matrix
    evals, evecs = np.linalg.eig(cov)

    # Sorts the eigenvalues, v1, [::-1] turns array around, 
    sort_indices = np.argsort(evals)[::-1]
    e_1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    e_2 = evecs[:, sort_indices[1]]

    # Theta to x_asix, arctan2 gives correct angle
    theta = np.arctan2(e_1[0], e_1[1])
  
    # Rotation, so that princple axis is vertical
    # anti-clockwise rotation matrix
    rotation = np.matrix([[np.cos(theta), -np.sin(theta)],
      [np.sin(theta), np.cos(theta)]])
    transformed_mat = rotation * np.stack([x,y])
    x_rot, y_rot = transformed_mat.A
  else: 
    x_rot, y_rot = x, y
  
  # Flipping
  n_flips = 0
  if flip:
    if weights[x_rot<0.].sum() < weights[x_rot>0.].sum():
      x_rot = -x_rot
      n_flips  += 1
    if weights[y_rot<0.].sum() > weights[y_rot>0.].sum():
      y_rot = -y_rot
      n_flips += 1
  
  return x_rot, y_rot


def mass (E,px,py,pz):

  mass = np.sqrt(np.maximum(0.,E**2-px**2-py**2-pz**2))
  return mass


def process_batch (start_id, input, output, n_constit, n_crop, intensity, rotate, flip, crop, norm, **kwargs):

    print_time("Loading input file (events %i to %i)"%(start_id,start_id+__batch_size__))
    df = pd.read_hdf(input,'table',start=start_id,stop=start_id+__batch_size__)
    if df.shape[0] == 0:
        return False
 
    print_time("Extracting 4-vectors")
    feat_list =  ["E","PX","PY","PZ"] 
    cols = ["{0}_{1}".format(feature,constit)
            for feature in feat_list for constit in range(n_constit)]
    vec4 = np.expand_dims(df[cols],axis=-1).reshape(-1, len(feat_list), n_constit)
    isig = df[__signal_col__]

    print_time("Calculating pT")
    E	  = vec4[:,0,:]
    pxs   = vec4[:,1,:]
    pys   = vec4[:,2,:]
    pzs   = vec4[:,3,:]
    pT    = np.sqrt(pxs**2+pys**2)

    print_time("Calculating eta")
    etas  = eta(pT,pzs)
    print_time("Calculating phi")
    phis  = phi(pxs,pys)

    print_time("Calculating the mass")
    E_tot = E.sum(axis=1)
    px_tot = pxs.sum(axis=1)
    py_tot = pys.sum(axis=1)
    pz_tot = pzs.sum(axis=1)
    j_mass = mass(E_tot, px_tot, py_tot, pz_tot)

    # Pre-shifting of phi
    phis = (phis.T - phis[:,0]).T
    phis[phis < -np.pi] += 2*np.pi
    phis[phis > np.pi] -= 2*np.pi
   
    print_time("Preprocessing")
    if intensity == "pT":
        weights = pT
    elif intensity == "E":
        weights = E

    for i in np.arange(0,__batch_size__):
        etas[i,:], phis[i,:] = preprocessing(etas[i,:], phis[i,:], weights[i,:], rotate, flip)

    # Using pT instead of energy E
    print_time("Creating images")
    z_ori = orig_image(etas, phis, pT)    
    print_time("Crop and normalize")
    z_new = np.zeros((z_ori.shape[0],n_crop, n_crop))
    
    for i in range(z_ori.shape[0]):
      if crop:
        Npix = z_ori[i,:,:].shape
        z_new[i,:,:] = z_ori[i, int(Npix[0]/2-n_crop/2):int(Npix[0]/2+n_crop/2), int(Npix[1]/2-n_crop/2):int(Npix[1]/2+n_crop/2)]
      else:
        z_new = z_ori
      if norm:
        z_sum = z_new[i,:,:].sum()
        if z_sum != 0.:
          z_new[i,:,:] = z_new[i,:,:]/z_sum

    print_time("Reshaping output")
    z_out = z_new.reshape((z_new.shape[0],-1))

    print_time("Creating output dataframe")
    out_cols = (["img_{0}".format(i) for i in range(z_new.shape[1]*z_new.shape[2])]
      + [__signal_col__] +[__mass_col__])

    df_out = pd.DataFrame(data=np.concatenate((z_out,isig[:,None],j_mass[:,None]),axis=1),
      index=np.arange(start_id,start_id+__batch_size__),
      columns=out_cols)
    print_time("Writing output file")
    df_out.to_hdf(output,"table",append=(start_id!=0),format="table",complib = "blosc", complevel=5)
     
    return True


def _run ():

  # Get input file from command line
  parser = argparse.ArgumentParser(description="Convert sequential input data to a calorimeter image")
  parser.add_argument("--input", type=str, required=True,
    help="Input file to convert")
  parser.add_argument("--output", type=str, required=True,
    help="Output file with calorimeter-based images")
  parser.add_argument("--n-events", type=int, default=-1,
    help="Number of events to convert [default `all`]")
  parser.add_argument("--n-constit", type=int, default=200,
    help="Upper limit of constitent in jet.")
  parser.add_argument("--n-crop", type=int, default=40,
    help="Number of pixels for final image.")
  parser.add_argument("--suffix", type=str, default="",
    help="Suffix to append to file name.")
  parser.add_argument("--intensity", type=str, default="pT", choices=["pT", "E"],
    help="Content in the image cells [transverse momentum (pT), or energy (E)]")
  parser.add_argument("--rotate", action="store_true",
    help="Rotate the image based on a PCA.")
  parser.add_argument("--flip", action="store_true",
    help="Flip the image such that the largest amount of energy is always on one side")
  parser.add_argument("--crop", action="store_true",
    help="Crop the image to desired size.")
  parser.add_argument("--norm", action="store_true",
    help="Norm the image by its total energy, i.e., the energy of the jet.")
  args = parser.parse_args()

  pprint.pprint(args.__dict__)
  max_batches = float(args.n_events)/__batch_size__
  start_id, n_shift_phi, n_shift_eta = 0, 0, 0
  # Start processing of data
  while process_batch(start_id, **args.__dict__):
    start_id += __batch_size__
    if start_id // __batch_size__ == max_batches:
      break

  print_time("Shuffling samples")
  df = pd.read_hdf(args.output,"table")
  df = df.iloc[np.random.permutation(len(df))]
  df.to_hdf(args.output, "table", format="table", complib = "blosc", complevel=5)
  
  if n_shift_eta != 0:
    print_time("Warning: hottest constituent is supposed to be close to origin.")
    print_time("Number of times eta of hottest const. was not close to origin: " + str(n_shift_eta))
  if n_shift_phi != 0:
    print_time("Warning: hottest constituent is supposed to be close to origin.")
    print_time("Number of times phi of hottest const. was not close to origin: " + str(n_shift_phi))
  print_time("Finished")


if __name__ == "__main__":
  
  _run()
