import os
import numpy as np
import tarfile
import numpy as np
import pytools as pt
import norm3
import argparse


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-d', '--dayside', action='store_true')
arg_parser.add_argument('-o', '--outdir', type=str)
args = arg_parser.parse_args()

try:
    os.makedirs(args.outdir)
except FileExistsError:
    pass

Re = 6371000 # Earth's radius

# Choose spatial domain
if args.dayside:
    xmin = -20
    xmax = 9
    zmin = -8
    zmax = 8
else:
    xmin = -30
    xmax = -5
    zmin = -5
    zmax = 5

boxre = [xmin, xmax, zmin, zmax]


for t in range(3600, 4200):
    # Loading x points for ground truth
    x_loc_dir = '/proj/ivanzait/x_points_calc/'
    x_loc_file = 'x_point_location_'+str(t)+'.txt'

    with open(x_loc_dir + x_loc_file) as f:
        lines = f.readlines()

    x_xp_array = []
    z_xp_array = []

    for k in range(len(lines)):       
        lineK = lines[k]    
        splitted = lineK.split(' ')
        x_xp = np.divide(float(splitted[0]), Re)
        z_xp = np.divide(float(splitted[2]), Re)
        x_xp_array.append(x_xp) # x position of Xpoint
        z_xp_array.append(z_xp) # z position of Xpoint


    # Selecting x points within the domain
    labeling_x = []
    labeling_z = []
    for kkk in range(0,np.array(x_xp_array).shape[0]):
        if x_xp_array[kkk] > xmin and x_xp_array[kkk] < xmax and z_xp_array[kkk] > zmin and z_xp_array[kkk] < zmax :
            labeling_x.append(x_xp_array[kkk])
            labeling_z.append(z_xp_array[kkk])
    
         
    # Reading the simulation data
    base_path = '/wrk/group/spacephysics/vlasiator/2D/BCH/'
    bulk_path = f'{base_path}/bulk/'
    file_name = f'bulk.000{str(t)}.vlsv'
    file = pt.vlsvfile.VlsvReader(bulk_path + file_name)
    
    
    # Normalization
    anisotropy, agyrotropy = norm3.normalization(
        filename=bulk_path + file_name, boxre=boxre,
        pass_vars=['anisotropy', 'agyrotropy']
    )
    
    B = norm3.masking(bulk_path + file_name, boxre, 'B')
    Bx = B[:,:,0]
    By = B[:,:,1]
    Bz = B[:,:,2]

    E = norm3.masking(bulk_path + file_name, boxre, 'E')
    Ex = E[:,:,0]
    Ey = E[:,:,1]
    Ez = E[:,:,2]

    rho = norm3.masking(bulk_path + file_name, boxre, 'rho')
    rho_v = norm3.masking(bulk_path + file_name, boxre, 'rho_v')

    v = rho_v / rho[:,:,np.newaxis]
    v[np.isnan(v)] = 0
    vx = v[:,:,0]
    vy = v[:,:,1]
    vz = v[:,:,2]

    # Creating 1D arrays of X and Z coordinates 
    xx = np.linspace(xmin, xmax, (np.array(B).shape[1]))
    zz = np.linspace(zmin, zmax, (np.array(B).shape[0]))

    # Labeling
    labeled_domain = np.zeros_like(B[:,:,0]) # Create zero matrix with size equal to spatial domain size
    for k in range(0,np.array(labeling_x).shape[0]): # Cycle over X-points
        idx_x = (np.abs(xx - np.array(labeling_x)[k])).argmin() # Column index of X-point pixel  
        idx_z = (np.abs(zz - np.array(labeling_z)[k])).argmin() # Row index of X-point pixel 
        labeled_domain[idx_z,idx_x] = 1 # Chabge 0 to 1 for the pixels with X-point
    
    # Storing   
    data = {
        'Bx': Bx, 'By': By, 'Bz': Bz, 'rho': rho, 'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'vx': vx, 'vy': vy, 'vz': vz,
        'agyrotropy':agyrotropy, 'anisotropy': anisotropy, 'labeled_domain': labeled_domain,
        'xmin': xmin,'xmax': xmax, 'zmin': zmin, 'zmax': zmax
    }

    np.savez(f'{args.outdir}/{t}.npz', **data)
    
    print(file_name, 'exported')
