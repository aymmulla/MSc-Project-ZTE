import h5py
import numpy as np


def h5tonum(file, showshape=False, load_traj=True, stripdims=True):
    with h5py.File(file) as f:
        d = f['data'][:]
        if stripdims:
            d = d[0, :, 0]
        if load_traj:
            traj = np.moveaxis(f['trajectory'][:], -1, 0)
            if showshape:
                print(d.shape)
                print(traj.shape)
            return d, traj
        else:
            if showshape:
                print(d.shape)
            return d


def h5tonum_interleaved(file, showshape=False, load_traj=True, stripdims=True):
    with h5py.File(file) as f:
        d = f['data'][:]
        if stripdims:
            d = d[0, 0]
        d = np.moveaxis(d,-1, 0)
        if load_traj:
            traj = f['trajectory'][:]
            if showshape:
                print(d.shape)
                print(traj.shape)
            return d, traj
        else:
            if showshape:
                print(d.shape)
            return d