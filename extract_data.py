import os
import numpy as np
import pytools as pt
import features
from utils import wrap, resize
from constants import Re, xmin, xmax, zmin, zmax, runs
import argparse


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-o', '--outdir', type=str)
arg_parser.add_argument('-x', '--x_dir', default='x_points', type=str)
args = arg_parser.parse_args()

try:
    os.makedirs(args.outdir)
except FileExistsError:
    pass

boxre = [xmin, xmax, zmin, zmax]

for run in runs:

    run_id = run['id']
    start_time = run['t_min']
    end_time = run['t_max']

    for t in range(start_time, end_time+1):
        # Loading x points
        x_loc_file = f'{args.x_dir}/{run_id}_x_points_{t}.txt'

        with open(x_loc_file) as f:
            lines = f.readlines()

        x_xp_array = []
        z_xp_array = []

        for k in range(len(lines)):
            lineK = lines[k]
            splitted = lineK.split(' ')
            x_xp = np.divide(float(splitted[0]), Re)
            z_xp = np.divide(float(splitted[2]), Re)
            x_xp_array.append(x_xp)  # x position of Xpoint
            z_xp_array.append(z_xp)  # z position of Xpoint

        # select x points within the domain
        labeling_x = []
        labeling_z = []
        for kkk in range(0, np.array(x_xp_array).shape[0]):
            if x_xp_array[kkk] > xmin and x_xp_array[kkk] < xmax and z_xp_array[kkk] > zmin and z_xp_array[kkk] < zmax:
                labeling_x.append(x_xp_array[kkk])
                labeling_z.append(z_xp_array[kkk])

        # read simulation data
        file_name = f'/wrk-vakka/group/spacephysics/vlasiator/2D/{run_id}/bulk/bulk.000{t}.vlsv'
        file = pt.vlsvfile.VlsvReader(file_name)

        B = features.masking(file_name, boxre, 'B')
        B_mag = np.linalg.norm(B, axis=-1)
        Bx = B[:, :, 0]
        By = B[:, :, 1]
        Bz = B[:, :, 2]

        E = features.masking(file_name, boxre, 'E')
        E_mag = np.linalg.norm(E, axis=-1)
        Ex = E[:, :, 0]
        Ey = E[:, :, 1]
        Ez = E[:, :, 2]
        
        v = features.masking(file_name, boxre, 'v')
        v_mag = np.linalg.norm(v, axis=-1)
        vx = v[:, :, 0]
        vy = v[:, :, 1]
        vz = v[:, :, 2]

        rhom = features.masking(file_name, boxre, 'rhom')
        rhoq = features.masking(file_name, boxre, 'rhoq')
        rho = features.masking(file_name, boxre, 'rho')
        
        pressure = features.masking(file_name, boxre, 'pressure')
        temperature = features.masking(file_name, boxre, 'temperature')
        beta = features.masking(file_name, boxre, 'beta')
        pdyn = features.masking(file_name, boxre, 'pdyn')

        # calculate pressure agyrotropy and anisotropy
        agyrotropy, anisotropy = features.normalization(
            filename=file_name, boxre=boxre,
            pass_vars=['agyrotropy', 'anisotropy']
        )

        # create 1D arrays of x and z coordinates
        xx = np.linspace(xmin, xmax, (np.array(B).shape[1]))
        zz = np.linspace(zmin, zmax, (np.array(B).shape[0]))

        # label reconnection
        reconnection = np.zeros_like(B[:, :, 0])  # create zero matrix with size equal to spatial domain size
        for k in range(0, np.array(labeling_x).shape[0]):  # cycle over X-points
            idx_x = (np.abs(xx - np.array(labeling_x)[k])).argmin()  # column index of X-point pixel
            idx_z = (np.abs(zz - np.array(labeling_z)[k])).argmin()  # row index of X-point pixel
            reconnection[idx_z, idx_x] = 1  # change 0 to 1 for the pixels with X-point
        reconnection = wrap(reconnection)

        # concatenate processed features
        frame_data = np.stack([B_mag, Bx, By, Bz, E_mag, Ex, Ey, Ez, v_mag, vx, vy, vz,
                               rho, rhom, rhoq, pressure, temperature, beta, pdyn,
                               agyrotropy, anisotropy, reconnection], axis=-1)
        
        np.save(f'{args.outdir}/{run_id}_{t}.npy', resize(frame_data))

        print(f'Extracted frame {run_id}_{t}')
