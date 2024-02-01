import os
import numpy as np
import pytools as pt
import features
from utils import wrap, resize
import argparse


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-o', '--outdir', type=str)
args = arg_parser.parse_args()

try:
    os.makedirs(args.outdir)
except FileExistsError:
    pass

Re = 6371000  # Earth's radius

xmin = -20
xmax = 10
zmin = -10
zmax = 10

boxre = [xmin, xmax, zmin, zmax]

num_frames = 5
step = 60
start_time = 3000
end_time = 4200

frame_count = 0
data_list = []

for t in range(start_time, end_time, step):
    # Loading x points for ground truth
    x_loc_dir = 'x_points/'
    x_loc_file = 'x_point_location_' + str(t) + '.txt'

    with open(x_loc_dir + x_loc_file) as f:
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
    base_path = '/wrk-vakka/group/spacephysics/vlasiator/2D/BCH/'
    bulk_path = f'{base_path}/bulk/'
    file_name = f'bulk.000{str(t)}.vlsv'
    file = pt.vlsvfile.VlsvReader(bulk_path + file_name)

    # normalization
    anisotropy, agyrotropy = features.normalization(
        filename=bulk_path + file_name, boxre=boxre,
        pass_vars=['anisotropy', 'agyrotropy']
    )

    B = features.masking(bulk_path + file_name, boxre, 'B')
    Bx = B[:, :, 0]
    By = B[:, :, 1]
    Bz = B[:, :, 2]

    E = features.masking(bulk_path + file_name, boxre, 'E')
    Ex = E[:, :, 0]
    Ey = E[:, :, 1]
    Ez = E[:, :, 2]

    v = features.masking(bulk_path + file_name, boxre, 'v')
    vx = v[:, :, 0]
    vy = v[:, :, 1]
    vz = v[:, :, 2]

    rho = features.masking(bulk_path + file_name, boxre, 'rho')

    rhom = features.masking(bulk_path + file_name, boxre, 'rhom')

    rhoq = features.masking(bulk_path + file_name, boxre, 'rhoq')

    beta = features.masking(bulk_path + file_name, boxre, 'beta')

    temperature = features.masking(bulk_path + file_name, boxre, 'temperature')

    pressure = features.masking(bulk_path + file_name, boxre, 'pressure')

    pdyn = features.masking(bulk_path + file_name, boxre, 'pdyn')


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
    frame_data = np.stack([Bx, By, Bz, Ex, Ey, Ez, vx, vy, vz, 
                           rho, rhom, rhoq, beta, temperature, pressure, pdyn,
                           agyrotropy, anisotropy, reconnection], axis=-1)
    
    data_list.append(resize(frame_data))

    frame_count += 1

    if frame_count == num_frames:
        # Stack all frame data
        all_data = np.stack(data_list, axis=0)
        np.save(f'{args.outdir}/frames_{t - frame_count + 1}_to_{t}.npy', all_data)

        print(f'Frames {t - frame_count + 1} to {t} exported')

        frame_count = 0
        data_list = []
