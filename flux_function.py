import os
import pytools as pt
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from shapely import geometry
from numpy import linalg as LA

# This script searches for the x and o points from the 2D simulations. It assumes polar plane.
# If you use equatorial plane change the z_array to y_array and it's limits.
# It uses the contours of grad(flux_function) to find the extrema and Hessian matrix to define the type of the point (minima, maxima, saddle)
# Note that for general use it defines the maxima as the o point.
# If you want to find the o-points, when they are the minima (e.g. BCV magnetopause reconnection), you have to change the sign, where the points are defined

# run id
run_id = "BCH"

# fluxfunction files
flux_file = "bulk."
if run_id == "BCQ":
    flux_file = "flux."

flux_path = "/wrk-vakka/group/spacephysics/vlasiator/2D/" + run_id + "/flux/" + flux_file
flux_suffix = ".bin"

# bulk files
bulk_path = "/wrk-vakka/group/spacephysics/vlasiator/2D/" + run_id + "/bulk/bulk."
bulk_suffix = ".vlsv"

bulkfile = bulk_path + str(3600).zfill(7) + bulk_suffix

path_to_save = '/proj/hdaniel/2d-reconnection-data/x_points'

os.makedirs(path_to_save, exist_ok=True)

RE = 6371000

# find intersection of two contours
def findIntersection(v1, v2):
    poly1 = geometry.LineString(v1)
    poly2 = geometry.LineString(v2)
    intersection = poly1.intersection(poly2)
    return intersection


# open bulkfile, determine sizes ##
vlsvfile = pt.vlsvfile.VlsvReader(bulkfile)
x_cells = int(vlsvfile.get_spatial_mesh_size()[0])
z_cells = int(vlsvfile.get_spatial_mesh_size()[2])
xsize = vlsvfile.read_parameter("xcells_ini")
xmax = vlsvfile.read_parameter("xmax")
xmin = vlsvfile.read_parameter("xmin")
zmin = vlsvfile.read_parameter("zmin")
zmax = vlsvfile.read_parameter("zmax")
dx = (xmax - xmin) / xsize

# define arrays for axes
x_array = np.array(range(int(xmin), int(xmax), int(dx)))
z_array = np.array(range(int(zmin), int(zmax), int(dx)))

for index in range(3000, 4200):
    fluxfile = flux_path + str(index).zfill(7) + flux_suffix

    # open input fluxfile
    flux_function = np.fromfile(fluxfile, dtype='double').reshape(z_cells, x_cells)
    flux_offset = float(index) * 0.3535 * 5e-9 * (-7.5e5)

    # smooth fluxfunction
    kernel_size = 5
    fkernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    raw_flux_function = flux_function
    flux_function = convolve2d(flux_function, fkernel, 'same')

    # calculate gradient of flux function
    dfdx, dfdz = np.gradient(flux_function)

    # calculate the 0 contours of df/dx and df/dz
    pl.figure(1)
    contour1 = plt.contour(x_array, z_array, dfdx, [0])
    contour1_paths = contour1.collections[0].get_paths()
    contour2 = plt.contour(x_array, z_array, dfdz, [0])
    contour2_paths = contour2.collections[0].get_paths()

    x_coords = []
    z_coords = []

    # find the intersection points of the
    for path1 in contour1_paths:
        for path2 in contour2_paths:
            if path1.intersects_path(path2) and len(path1) > 1 and len(path2) > 1:
                intersection = findIntersection(path1.vertices, path2.vertices)
                if intersection.is_empty:
                    continue
                elif isinstance(intersection, geometry.Point):
                    x_coords.append(intersection.x)
                    z_coords.append(intersection.y)
                elif isinstance(intersection, geometry.MultiPoint):
                    for point in intersection.geoms:
                        x_coords.append(point.x)
                        z_coords.append(point.y)

    # define the type of the gradient(flux) = 0
    x_point_location = []
    o_point_location = []
    x_point_fluxes = []
    o_point_fluxes = []
    minima_location = []
    flux_function = flux_function.T

    for k in range(len(x_coords)):
        # cellid = 1+i+j*x_cells
        coords = [x_coords[k], 0, z_coords[k]]
        cellid = vlsvfile.get_cellid(coords)
        i = int((cellid - 1) % x_cells)
        j = (int(cellid) - 1) // x_cells

        difference = []

        # Hessian matrix using central difference formulas for the second partial derivatives
        deltaPsi_xx = (flux_function[i + 1, j] - 2 * flux_function[i, j] + flux_function[i - 1, j]) / dx**2
        deltaPsi_zz = (flux_function[i, j + 1] - 2 * flux_function[i, j] + flux_function[i, j - 1]) / dx**2
        deltaPsi_xz = (flux_function[i + 1, j + 1] - flux_function[i + 1, j - 1] - flux_function[i - 1, j + 1] + flux_function[i - 1, j - 1]) / (4 * dx**2)

        Hessian = [[deltaPsi_xx, deltaPsi_xz], [deltaPsi_xz, deltaPsi_zz]]

        if not np.isfinite(Hessian).all():
            continue

        DetHess = deltaPsi_xx * deltaPsi_zz - deltaPsi_xz * deltaPsi_xz
        eigvals, eigvectors = LA.eig(Hessian)

        # Calculate interpolated flux function value
        i_i = int(coords[0] / dx)
        i_f = coords[0] / dx - i_i
        j_i = int(coords[1] / dx)
        j_f = coords[1]
        interpolated_flux = (1. - j_f) * ((1. - i_f) * flux_function[i, j] + i_f * flux_function[i + 1, j]) + j_f * ((1. - i_f) * flux_function[i, j + 1] + i_f * flux_function[i + 1, j + 1])

        if DetHess < 0:
            x_point_location.append(coords)
            x_point_fluxes.append(interpolated_flux)

        # if you want the o-points to be local maxima use deltaPsi_xx < 0
        # if you want them to be local minima use  deltaPsi_xx > 0
        # if DetHess > 0 and deltaPsi_xx > 0:
        #     minima_location.append(coords)
        if DetHess > 0 and deltaPsi_xx < 0:
            o_point_location.append(coords)
            o_point_fluxes.append(interpolated_flux)

    pl.close('all')

    np.savetxt(path_to_save + "/o_point_location_" + str(index) + ".txt", o_point_location)
    np.savetxt(path_to_save + "/o_point_location_and_fluxes_" + str(index) + ".txt",
               np.concatenate((o_point_location, np.array(o_point_fluxes)[:, np.newaxis]), axis=1))

    np.savetxt(path_to_save + "/x_point_location_" + str(index) + ".txt", x_point_location)
    np.savetxt(path_to_save + "/x_point_location_and_fluxes_" + str(index) + ".txt",
               np.concatenate((x_point_location, np.array(x_point_fluxes)[:, np.newaxis]), axis=1))
