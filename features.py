import scipy
import numpy as np
import pytools as pt
import reduction


def masking(filename, boxre, mapval):
    f = pt.vlsvfile.VlsvReader(filename)
    Re = 6.371e+6  # Earth radius in m

    [xsize, ysize, zsize] = f.get_spatial_mesh_size()
    xsize = int(xsize)
    ysize = int(ysize)
    zsize = int(zsize)
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    cellsize = (xmax - xmin) / xsize
    cellids = f.read_variable("CellID")
    pt.plot.plot_helpers.CELLSIZE = cellsize

    simext = [xmin, xmax, zmin, zmax]
    sizes = [xsize, zsize]

    if len(boxre) == 4:
        boxcoords = [i * Re for i in boxre]
    else:
        boxcoords = list(simext)

    # Select window to draw
    # boxcoords=list(simext)
    # If box extents were provided manually, truncate to simulation extents
    boxcoords[0] = max(boxcoords[0], simext[0])
    boxcoords[1] = min(boxcoords[1], simext[1])
    boxcoords[2] = max(boxcoords[2], simext[2])
    boxcoords[3] = min(boxcoords[3], simext[3])

    # Scale data extent and plot box
    axisunit = Re
    simext = [i / axisunit for i in simext]
    boxcoords = [i / axisunit for i in boxcoords]

    # MESH   MESH   MESH   MESH
    # Generates the mesh to map the data to.
    [XmeshXY, YmeshXY] = scipy.meshgrid(np.linspace(simext[0], simext[1], num=sizes[0] + 1), 
                                        np.linspace(simext[2], simext[3], num=sizes[1] + 1))

    # The grid generated by meshgrid has all four corners for each cell.
    # We mask using only the centre values.
    # Calculate offsets for cell-centre coordinates
    XmeshCentres = XmeshXY[:-1, :-1] + 0.5 * (XmeshXY[0, 1] - XmeshXY[0, 0])
    YmeshCentres = YmeshXY[:-1, :-1] + 0.5 * (YmeshXY[1, 0] - YmeshXY[0, 0])
    maskgrid = np.ma.array(XmeshCentres)

    pass_full = None
    if not pass_full:
        # If zoomed-in using a defined box, and not specifically asking to pass all values:
        # Generate mask for only visible section (with small buffer for e.g. gradient calculations)
        maskboundarybuffer = 2. * cellsize / axisunit
        maskgrid = np.ma.masked_where(XmeshCentres < (boxcoords[0] - maskboundarybuffer), maskgrid)
        maskgrid = np.ma.masked_where(XmeshCentres > (boxcoords[1] + maskboundarybuffer), maskgrid)
        maskgrid = np.ma.masked_where(YmeshCentres < (boxcoords[2] - maskboundarybuffer), maskgrid)
        maskgrid = np.ma.masked_where(YmeshCentres > (boxcoords[3] + maskboundarybuffer), maskgrid)

    if np.ma.is_masked(maskgrid):
        # Save lists for masking
        MaskX = np.where(~np.all(maskgrid.mask, axis=1))[0]  # [0] takes the first element of a tuple
        MaskY = np.where(~np.all(maskgrid.mask, axis=0))[0]
        XmeshPass = XmeshXY[MaskX[0]:MaskX[-1] + 2, :]
        XmeshPass = XmeshPass[:, MaskY[0]:MaskY[-1] + 2]
        YmeshPass = YmeshXY[MaskX[0]:MaskX[-1] + 2, :]
        YmeshPass = YmeshPass[:, MaskY[0]:MaskY[-1] + 2]
        XmeshCentres = XmeshCentres[MaskX[0]:MaskX[-1] + 1, :]
        XmeshCentres = XmeshCentres[:, MaskY[0]:MaskY[-1] + 1]
        YmeshCentres = YmeshCentres[MaskX[0]:MaskX[-1] + 1, :]
        YmeshCentres = YmeshCentres[:, MaskY[0]:MaskY[-1] + 1]
    else:
        XmeshPass = np.ma.array(XmeshXY)
        YmeshPass = np.ma.array(YmeshXY)

    pass_map = f.read_variable(mapval)

    if np.ndim(pass_map) == 1:
        pass_map = pass_map[cellids.argsort()].reshape([sizes[1], sizes[0]])
    elif np.ndim(pass_map) == 2:  # vector variable
        pass_map = pass_map[cellids.argsort()].reshape([sizes[1], sizes[0], pass_map.shape[1]])
    elif np.ndim(pass_map) == 3:  # tensor variable
        pass_map = pass_map[cellids.argsort()].reshape([sizes[1], sizes[0], pass_map.shape[1], pass_map.shape[2]])

    if np.ma.is_masked(maskgrid):
        if np.ndim(pass_map) == 2:
            pass_map = pass_map[MaskX[0]:MaskX[-1] + 1, :]
            pass_map = pass_map[:, MaskY[0]:MaskY[-1] + 1]
        elif np.ndim(pass_map) == 3:  # vector variable
            pass_map = pass_map[MaskX[0]:MaskX[-1] + 1, :, :]
            pass_map = pass_map[:, MaskY[0]:MaskY[-1] + 1, :]
        elif np.ndim(pass_map) == 4:  # tensor variable
            pass_map = pass_map[MaskX[0]:MaskX[-1] + 1, :, :, :]
            pass_map = pass_map[:, MaskY[0]:MaskY[-1] + 1, :, :]
        else:
            print("Error in masking pass_maps!")

    return pass_map


def masking_val(filename, boxre, value):
    f = pt.vlsvfile.VlsvReader(filename)
    Re = 6.371e+6  # Earth radius in m
    
    [xsize, ysize, zsize] = f.get_spatial_mesh_size()
    xsize = int(xsize)
    ysize = int(ysize)
    zsize = int(zsize)
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    cellsize = (xmax - xmin) / xsize
    cellids = f.read_variable("CellID")
    pt.plot.plot_helpers.CELLSIZE = cellsize

    simext = [xmin, xmax, zmin, zmax]
    sizes = [xsize, zsize]

    if len(boxre) == 4:
        boxcoords = [i * Re for i in boxre]
    else:
        boxcoords = list(simext)

    # Select window to draw
    # boxcoords=list(simext)
    # If box extents were provided manually, truncate to simulation extents
    boxcoords[0] = max(boxcoords[0], simext[0])
    boxcoords[1] = min(boxcoords[1], simext[1])
    boxcoords[2] = max(boxcoords[2], simext[2])
    boxcoords[3] = min(boxcoords[3], simext[3])

    # Scale data extent and plot box
    axisunit = Re
    simext = [i / axisunit for i in simext]
    boxcoords = [i / axisunit for i in boxcoords]

    # MESH   MESH   MESH   MESH
    # Generates the mesh to map the data to.
    [XmeshXY, YmeshXY] = scipy.meshgrid(np.linspace(simext[0], simext[1], num=sizes[0] + 1), 
                                        np.linspace(simext[2], simext[3], num=sizes[1] + 1))

    # The grid generated by meshgrid has all four corners for each cell.
    # We mask using only the centre values.
    # Calculate offsets for cell-centre coordinates
    XmeshCentres = XmeshXY[:-1, :-1] + 0.5 * (XmeshXY[0, 1] - XmeshXY[0, 0])
    YmeshCentres = YmeshXY[:-1, :-1] + 0.5 * (YmeshXY[1, 0] - YmeshXY[0, 0])
    maskgrid = np.ma.array(XmeshCentres)

    pass_full = None
    if not pass_full:
        # If zoomed-in using a defined box, and not specifically asking to pass all values:
        # Generate mask for only visible section (with small buffer for e.g. gradient calculations)
        maskboundarybuffer = 2. * cellsize / axisunit
        maskgrid = np.ma.masked_where(XmeshCentres < (boxcoords[0] - maskboundarybuffer), maskgrid)
        maskgrid = np.ma.masked_where(XmeshCentres > (boxcoords[1] + maskboundarybuffer), maskgrid)
        maskgrid = np.ma.masked_where(YmeshCentres < (boxcoords[2] - maskboundarybuffer), maskgrid)
        maskgrid = np.ma.masked_where(YmeshCentres > (boxcoords[3] + maskboundarybuffer), maskgrid)

    if np.ma.is_masked(maskgrid):
        # Save lists for masking
        MaskX = np.where(~np.all(maskgrid.mask, axis=1))[0]  # [0] takes the first element of a tuple
        MaskY = np.where(~np.all(maskgrid.mask, axis=0))[0]
        XmeshPass = XmeshXY[MaskX[0]:MaskX[-1] + 2, :]
        XmeshPass = XmeshPass[:, MaskY[0]:MaskY[-1] + 2]
        YmeshPass = YmeshXY[MaskX[0]:MaskX[-1] + 2, :]
        YmeshPass = YmeshPass[:, MaskY[0]:MaskY[-1] + 2]
        XmeshCentres = XmeshCentres[MaskX[0]:MaskX[-1] + 1, :]
        XmeshCentres = XmeshCentres[:, MaskY[0]:MaskY[-1] + 1]
        YmeshCentres = YmeshCentres[MaskX[0]:MaskX[-1] + 1, :]
        YmeshCentres = YmeshCentres[:, MaskY[0]:MaskY[-1] + 1]
    else:
        XmeshPass = np.ma.array(XmeshXY)
        YmeshPass = np.ma.array(YmeshXY)

    pass_map = value

    if np.ndim(pass_map) == 1:
        pass_map = pass_map[cellids.argsort()].reshape([sizes[1], sizes[0]])
    elif np.ndim(pass_map) == 2:  # vector variable
        pass_map = pass_map[cellids.argsort()].reshape([sizes[1], sizes[0], pass_map.shape[1]])
    elif np.ndim(pass_map) == 3:  # tensor variable
        pass_map = pass_map[cellids.argsort()].reshape([sizes[1], sizes[0], pass_map.shape[1], pass_map.shape[2]])

    if np.ma.is_masked(maskgrid):
        if np.ndim(pass_map) == 2:
            pass_map = pass_map[MaskX[0]:MaskX[-1] + 1, :]
            pass_map = pass_map[:, MaskY[0]:MaskY[-1] + 1]
        elif np.ndim(pass_map) == 3:  # vector variable
            pass_map = pass_map[MaskX[0]:MaskX[-1] + 1, :, :]
            pass_map = pass_map[:, MaskY[0]:MaskY[-1] + 1, :]
        elif np.ndim(pass_map) == 4:  # tensor variable
            pass_map = pass_map[MaskX[0]:MaskX[-1] + 1, :, :, :]
            pass_map = pass_map[:, MaskY[0]:MaskY[-1] + 1, :, :]
        else:
            print("Error in masking pass_maps!")

    return pass_map


def normalization(filename, boxre, pass_vars):
    f = pt.vlsvfile.VlsvReader(filename)
    massive = []
    for mapval in pass_vars:
        if mapval == 'B':
            pass_map = masking(filename, boxre, mapval)
            absB = np.sqrt(pass_map[:, :, 0]**2 + pass_map[:, :, 1]**2 + pass_map[:, :, 2]**2)
            Bmax = np.amax(absB)
            pass_map = np.divide(pass_map, Bmax)
            B_norm = pass_map
            massive.append(B_norm)

        if mapval == 'rho':
            pass_map = masking(filename, boxre, mapval)
            rhomax = np.amax(pass_map)
            pass_map = np.divide(pass_map, rhomax)
            rho_norm = pass_map
            massive.append(rho_norm)
            mu0 = 1.25663706144e-6
            me = 9.10938356e-31
            mi = 1836 * me
            Va = Bmax / np.sqrt(mu0 * mi * rhomax)

        if mapval == 'rho_v':
            rho = masking(filename, boxre, 'rho')
            rho_V_masked = masking(filename, boxre, mapval)
            v = np.empty((rho_V_masked.shape[0], rho_V_masked.shape[1], 3))
            for i in range(rho_V_masked.shape[2]):
                VV = np.divide(rho_V_masked[:, :, i], rho)
                v[:, :, i] = VV
            V_norm = np.divide(v, Va)
            massive.append(V_norm)

        if mapval == 'v':
            rho = masking(filename, boxre, 'rho')
            rho_v_masked = masking(filename, boxre, 'rho_v')
            v = rho_v_masked / rho[:, :, np.newaxis]
            abs_v = np.sqrt(v[:, :, 0]**2 + v[:, :, 1]**2 + v[:, :, 2]**2)
            v_max = np.max(abs_v)
            v_norm = v / v_max
            massive.append(v_norm)

        if mapval == 'J':
            B = masking(filename, boxre, 'B')
            J = pt.plot.plot_helpers.vec_currentdensity(B)
            elementary_charge = 1.6 * 1e-19
            J0 = rhomax * elementary_charge * Va
            Jnorm = np.divide(J, J0)
            massive.append(Jnorm)

        if mapval == 'PTensorDiagonal':
            Ptensor = masking(filename, boxre, mapval)
            me = 9.10938356e-31
            mi = 1836 * me
            P0 = 0.5 * rhomax * mi * Va**2
            Pnorm = np.divide(Ptensor, P0)
            massive.append(Pnorm)

        if mapval == 'Slippage':
            E = masking(filename, boxre, 'E')
            B = masking(filename, boxre, 'B')
            # V = masking(filename,boxre,'V')
            rho_V_masked = masking(filename, boxre, 'rho_v')
            rho = masking(filename, boxre, 'rho')
            V = np.empty((rho_V_masked.shape[0], rho_V_masked.shape[1], 3))
            for i in range(rho_V_masked.shape[2]):
                VV = np.divide(rho_V_masked[:, :, i], rho)
                V[:, :, i] = VV
            Vperp = pt.plot.plot_helpers.VectorArrayPerpendicularVector(V, B)
            EcrossB = np.divide(np.cross(E, B), (B * B).sum(-1)[:, :, np.newaxis])
            metricSlippage = EcrossB - Vperp
            Slippage_norm = np.divide(metricSlippage, Va)
            massive.append(Slippage_norm)

        if mapval == 'EminusVxB':
            E = masking(filename, boxre, 'E')
            B = masking(filename, boxre, 'B')
            # V = masking(filename,boxre,'V')
            E0 = Bmax * Va

            Enorm = np.divide(E, E0)
            Bnorm = np.divide(B, Bmax)

            rho_V_masked = masking(filename, boxre, 'rho_v')
            rho = masking(filename, boxre, 'rho')
            V = np.empty((rho_V_masked.shape[0], rho_V_masked.shape[1], 3))
            for i in range(rho_V_masked.shape[2]):
                VV = np.divide(rho_V_masked[:, :, i], rho)
                V[:, :, i] = VV
            # Vperp = pt.plot.plot_helpers.VectorArrayPerpendicularVector(V,B)
            Vnorm = np.divide(V, Va)
            VcrossB_norm = np.cross(Vnorm, Bnorm)

            # EcrossB = np.divide(np.cross(E,B), (B*B).sum(-1)[:,:,np.newaxis])
            # metricSlippage = EcrossB-Vperp
            EminusVxB = Enorm - VcrossB_norm
            # EminusVxBtotal = np.sqrt(EminusVxB[:, :, 0]**2 + EminusVxB[:, :, 1]**2 + EminusVxB[:, :, 2]**2)
            massive.append(EminusVxB)

        if mapval == 'anisotropy':
            B = f.read_variable('B')
            Pdiag = f.read_variable('PTensorDiagonal')
            P0diag = f.read_variable('PTensorOffDiagonal')
            PTensor = reduction.FullTensor([Pdiag, P0diag])
            PTensor_rotated = reduction.RotatedTensor([PTensor, B])
            Ppar = reduction.ParallelTensorComponent([PTensor_rotated])
            Ppar = masking_val(filename, boxre, Ppar)
            Pperp = reduction.PerpendicularTensorComponent([PTensor_rotated])
            Pperp = masking_val(filename, boxre, Pperp)

            B = masking(filename, boxre, 'B')
            absB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
            rho = masking(filename, boxre, 'rho')
            Bmax = np.max(absB)
            rhomax = np.max(rho)

            mu0 = 1.25663706144e-6
            me = 9.10938356e-31
            mi = 1836 * me

            Va = Bmax / np.sqrt(mu0 * mi * rhomax)
            P0 = 0.5 * rhomax * mi * Va**2

            Ppar = np.divide(Ppar, P0)
            Pperp = np.divide(Pperp, P0)
            anisotropy = np.divide(abs(Ppar - Pperp), (Pperp + Ppar))
            # Paniso = np.divide(Ppar, Pperp)

            anisotropy[np.isnan(anisotropy)] = 0

            massive.append(anisotropy)

        if mapval == 'EJ':
            Emap = masking(filename, boxre, 'E')
            Bmap = masking(filename, boxre, 'B')
            Bmap = np.divide(Bmap, Bmax)
            Jmap = pt.plot.plot_helpers.vec_currentdensity(Bmap)
            elementary_charge = 1.6 * 1e-19
            E0 = Bmax * Va
            Emap = np.divide(Emap, E0)
            nx = Jmap.shape[0]
            ny = Jmap.shape[1]
            EJ = np.zeros([nx, ny, 3])
            EJ[:, :, 0] = np.multiply(Jmap[:, :, 0], Bmap[:, :, 0])
            EJ[:, :, 1] = np.multiply(Jmap[:, :, 1], Bmap[:, :, 1])
            EJ[:, :, 2] = np.multiply(Jmap[:, :, 2], Bmap[:, :, 2])
            EJ_total = np.sqrt(EJ[:, :, 0]**2 + EJ[:, :, 1]**2 + EJ[:, :, 2]**2)
            massive.append(EJ_total)

        if mapval == 'agyrotropy':
            B = f.read_variable('B')
            Pdiag = f.read_variable('PTensorDiagonal')
            P0diag = f.read_variable('PTensorOffDiagonal')
            PTensor = reduction.FullTensor([Pdiag, P0diag])
            PTensor_rotated = reduction.RotatedTensor([PTensor, B])
            aGyrotropy = reduction.aGyrotropy([PTensor_rotated])
            aGyrotropy = masking_val(filename, boxre, aGyrotropy)
            massive.append(aGyrotropy)

        if mapval == 'E':
            Emap = masking(filename, boxre, 'E')

            # E0 = Bmax*Va
            # Emap = np.divide(Emap,E0)
            # E = np.sqrt(Emap[:, :, 0]**2 + Emap[:, :, 1]**2 + Emap[:, :, 2]**2)

            absE = np.sqrt(Emap[:, :, 0]**2 + Emap[:, :, 1]**2 + Emap[:, :, 2]**2)
            Emax = np.amax(absE)
            E = np.divide(Emap, Emax)

            massive.append(E)

    return massive
