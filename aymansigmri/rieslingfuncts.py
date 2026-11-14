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

def interleaved_reshape(kspace, trajectory, divis = 257):
    cleaned_kspace = []
    assert kspace.shape[1] == trajectory.shape[0], "mismatch in kspace/trajectory"
    if kspace.shape[1] % divis != 0:
        raise ValueError(f'{kspace.shape[1]} spokes not divisible by {divis}')
    for coil in kspace:
        num_segments = coil.shape[0] // divis
        segments = np.split(coil, num_segments)[:-4]
        cleaned_kspace.append(np.concatenate([s[1:] for s in segments], axis=0))

    cleaned_kspace = np.stack(cleaned_kspace)

    num_segments_traj = trajectory.shape[0] // divis
    segments_traj = np.split(trajectory, num_segments_traj)[:-4]
    cleaned_trajectory = np.concatenate([st[1:] for st in segments_traj], axis=0)

    return (cleaned_kspace, cleaned_trajectory)