import os
import sys 
# Add the directory containing the module to sys.path
module_dir = '/proj/ivanzait/analysator/'
sys.path.append(module_dir)
import pytools as pt
import numpy as np
import features
import reduction
from utils import wrap, resize
from constants import Re, xmin, xmax, zmin, zmax, runs
import argparse


def label_reconnection(labeling_x,labeling_z,B,boxre):
    # create 1D arrays of x and z coordinates
    [xmin, xmax,zmin, zmax] = boxre
    xx = np.linspace(xmin, xmax, (np.array(B).shape[1]))
    zz = np.linspace(zmin, zmax, (np.array(B).shape[0]))

    # label reconnection
    reconnection = np.zeros_like(B[:, :, 0])  # create zero matrix with size equal to spatial domain size
    for k in range(0, np.array(labeling_x).shape[0]):  # cycle over X-points
        idx_x = (np.abs(xx - np.array(labeling_x)[k])).argmin()  # column index of X-point pixel
        idx_z = (np.abs(zz - np.array(labeling_z)[k])).argmin()  # row index of X-point pixel
        reconnection[idx_z, idx_x] = 1  # change 0 to 1 for the pixels with X-point
    reconnection = wrap(reconnection)
    return reconnection


def get_agyrotropy(file_name,boxre,name_list):
    
    [E_name,B_name,rho_name,V_name,Pd_name,Pod_name]=name_list
    
    B=get_var(file_name, boxre, var_name=B_name, grid_flag='vg')       
    Pdiag = get_var(file_name, boxre, var_name=Pd_name, grid_flag='vg')       
    P0diag = get_var(file_name, boxre, var_name=Pod_name, grid_flag='vg')       

    sx,sy=B.shape[1],B.shape[0]
    
    Pdiag_flat=Pdiag.reshape(sx*sy,3)
    P0diag_flat=P0diag.reshape(sx*sy,3)    
    B_flat=B.reshape(sx*sy,3)
    
    PTensor = reduction.FullTensor([Pdiag_flat, P0diag_flat])        
    PTensor_rotated = reduction.RotatedTensor([PTensor, B_flat])

    aGyrotropy = reduction.aGyrotropy([PTensor_rotated])
    aGyrotropy=aGyrotropy.reshape(sy,sx)

    return aGyrotropy



def get_anisotropy(file_name,boxre,name_list):

    [E_name,B_name,rho_name,V_name,Pd_name,Pod_name]=name_list        

    B=get_var(file_name, boxre, var_name=B_name, grid_flag='vg')       
    Pdiag = get_var(file_name, boxre, var_name=Pd_name, grid_flag='vg')       
    P0diag = get_var(file_name, boxre, var_name=Pod_name, grid_flag='vg')       
    
    sx,sy=B.shape[1],B.shape[0]

    Pdiag_flat=Pdiag.reshape(sx*sy,3)
    P0diag_flat=P0diag.reshape(sx*sy,3)
    B_flat=B.reshape(sx*sy,3)
    
    PTensor = reduction.FullTensor([Pdiag_flat, P0diag_flat])    
    PTensor_rotated = reduction.RotatedTensor([PTensor, B_flat])
    
    Ppar = reduction.ParallelTensorComponent([PTensor_rotated])
    Pperp = reduction.PerpendicularTensorComponent([PTensor_rotated])    
    anisotropy = np.divide(abs(Ppar - Pperp), (Pperp + Ppar))

    anisotropy=anisotropy.reshape(sy,sx)    
    return anisotropy
    

def get_var(file_name, boxre, var_name, grid_flag):

    RE=6371000

    f=pt.vlsvfile.VlasiatorReader(file_name)
    #time=np.float32(f.read_parameter("time"))
    size=np.float32(f.get_fsgrid_mesh_size())
    nx=int(size[0])
    nz=int(size[2])
    #print(nx,nz)

    extents=np.float32(f.get_fsgrid_mesh_extent())
    #print(extents)
    cellsize=(extents[3]-extents[0])/nx
    #print('cellsize',cellsize)

    if grid_flag=='fg':
        var=np.float32(f.read_fsgrid_variable(var_name))
    elif grid_flag=='vg':
        var=np.float32(f.read_variable(var_name))

    if var.ndim==1:
        var=var.reshape(nx,nz)
    elif var.ndim==2:
        var=var.reshape(nx,nz,3)

    xlim_left=boxre[0]*RE
    xlim_right=boxre[1]*RE
    zlim_bottom=boxre[2]*RE
    zlim_top=boxre[3]*RE
        
    x_ind_left=int(abs((extents[0]-xlim_left)/cellsize))
    x_ind_right=int(abs((extents[0]-xlim_right)/cellsize))
    z_ind_bottom=int(abs((extents[2]-zlim_bottom)/cellsize))
    z_ind_top=int(abs((extents[2]-zlim_top)/cellsize))
    
    if var.ndim==3:
        var=var[x_ind_left:x_ind_right, z_ind_bottom:z_ind_top,:]
    elif var.ndim==2:
        var=var[x_ind_left:x_ind_right, z_ind_bottom:z_ind_top]

    return var

########################
######## MAIN C#########
########################        

#E_name='fg_e'
#B_name='vg_b_vol'

## DEFINE THE BOX
xmin, xmax, zmin, zmax=[-20,10,-10,10]
boxre = [xmin, xmax, zmin, zmax]

#t=300
#bulk_path='/wrk-vakka/group/spacephysics/vlasiator/2D/BGF/extendvspace_restart229/bulk/'
#file_name=bulk_path+'bulk.'+str(t).zfill(7)+'.vlsv'


############################################################
############################################################
############################################################

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-o', '--outdir', type=str)
arg_parser.add_argument('-x', '--x_dir', default='x_points', type=str)
args = arg_parser.parse_args()

try:
    os.makedirs(args.outdir)
except FileExistsError:
    pass

for run in runs:

    run_id = run['id']
    start_time = run['t_min']
    end_time = run['t_max']

    if run_id=='BCH':
        bulk_path='/wrk-vakka/group/spacephysics/vlasiator/2D/BCH/bulk/'
        x_dir='/wrk-vakka/group/spacephysics/vlasiator/2D/BCH/x_and_o_points/' 
        ### naming:       
        E_name='E'
        B_name='B'         
        rho_name='rho'
        V_name='rho_v'
        Pd_name ='PTensorDiagonal'
        Pod_name = 'PTensorOffDiagonal'     
        
    if run_id=='BGF':
        bulk_path='/wrk-vakka/group/spacephysics/vlasiator/2D/BGF/extendvspace_restart229/bulk/'
        x_dir='/wrk-vakka/group/spacephysics/vlasiator/2D/BGF/ivan/x_and_o_points/'        
        ### naming:        
        E_name='fg_e'
        B_name='vg_b_vol'
        rho_name='proton/vg_rho'
        V_name='proton/vg_v'        
        Pd_name ='proton/vg_ptensor_diagonal'
        Pod_name = 'proton/vg_ptensor_offdiagonal'     


    name_list=[E_name,B_name,rho_name,V_name,Pd_name,Pod_name]

    for t in range(start_time, end_time+1):
        print('run',run_id,'t=',str(t))                
        
        # Loading x points
        # x_loc_file = f'{args.x_dir}/{run_id}_x_points_{t}.txt'
        x_loc_file = x_dir+'x_point_location_'+str(t)+'.txt'
        labeling_x,labeling_z=features.get_x_points(x_loc_file,boxre)
        #print('x points are ready')

#         # read simulation data
#         #file_name = f'/wrk-vakka/group/spacephysics/vlasiator/2D/{run_id}/bulk/bulk.{str(t).zfill(7)}.vlsv'
        file_name=bulk_path+'bulk.'+str(t).zfill(7)+'.vlsv'
        #file = pt.vlsvfile.VlsvReader(file_name)

        #B = features.masking(file_name, boxre, B_name)
        B = get_var(file_name, boxre, B_name,grid_flag='vg')        
        B_mag = np.linalg.norm(B, axis=-1)
        Bx,By,Bz = B[:, :, 0],B[:, :, 1],B[:, :, 2]
        print('B done')

        #E = features.masking(file_name, boxre, E_name)
        if run_id=='BCH':
            E = get_var(file_name, boxre, E_name,grid_flag='vg')
        elif run_id=='BGF':
            E = get_var(file_name, boxre, E_name,grid_flag='fg')                
        E_mag = np.linalg.norm(E, axis=-1)
        Ex,Ey,Ez = B[:, :, 0], E[:, :, 1], E[:, :, 2]
        print('E done')

        #rho = features.masking(file_name, boxre, rho_name)
        rho = get_var(file_name, boxre, rho_name,grid_flag='vg')
        earth_mask = rho == 0
        print('rho done')

        #v = features.masking(file_name, boxre, V_name)
        v = get_var(file_name, boxre, V_name,grid_flag='vg')
        v_mag = np.linalg.norm(v, axis=-1)
        vx,vy,vz = v[:, :, 0],v[:, :, 1],v[:, :, 2]
        print('v done')
      
        # calculate pressure agyrotropy and anisotropy
        anisotropy = features.get_anisotropy(file_name=file_name, boxre=boxre,name_list=name_list)
        agyrotropy = features.get_agyrotropy(file_name=file_name, boxre=boxre,name_list=name_list)
        reconnection = label_reconnection(labeling_x,labeling_z,B,boxre)

        # concatenate processed features
#        frame_data = np.stack([B_mag, Bx, By, Bz, E_mag, Ex, Ey, Ez, v_mag, vx, vy, vz,
#                               rho, agyrotropy, anisotropy, reconnection], axis=-1)

        frame_data = np.stack([B_mag, Bx, By, Bz, E_mag, Ex, Ey, Ez, v_mag, vx, vy, vz,
                               rho,agyrotropy,anisotropy,reconnection], axis=-1)

        frame_data[earth_mask] = 0

        np.save(f'{args.outdir}/{run_id}_{t}.npy', resize(frame_data))

        print(f'Extracted frame {run_id}_{t}')
