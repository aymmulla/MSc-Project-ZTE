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






'''zte_clean = f'{zte}_clean'

data_out = np.moveaxis(cleaned_data, 0, -1)[None, None].astype(np.complex64)  # (1, 1, 64000, 248, 18)
traj_out = cleaned_trajectory.astype(np.float32)                              # (64000, 248, 3)
assert data_out.shape[2:4] == traj_out.shape[:2]

with h5py.File(f'{zte}.h5', 'r') as src, h5py.File(f'{zte_clean}.h5', 'w') as out:
    d = out.create_dataset('data', data=data_out, chunks=True, compression='gzip')
    for i, label in enumerate(['t', 'slab', 'trace', 'sample', 'channel']):
        d.dims[i].label = label

    t = out.create_dataset('trajectory', data=traj_out, compression='gzip')
    matrix = np.asarray(src['trajectory'].attrs['matrix']).reshape(-1)
    t.attrs.create('matrix', matrix, dtype=h5py.h5t.array_create(h5py.h5t.NATIVE_INT64, (3,)))

    src.copy('info', out)  # keep voxel size / origin / direction / tr

print(data_out.shape, traj_out.shape)'''