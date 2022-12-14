from contextlib import contextmanager
import os
import shutil
import sys

import asdf
import asdf_zarr
import zarr


if __name__ == '__main__':
    fn = 'test_internal_zarr.asdf'

    # create the zarr array in a temporary location
    tmp_dir = 'tmp'
    store = zarr.storage.DirectoryStore(tmp_dir)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    my_zarr = zarr.create(
        store=store, shape=100, chunks=10, dtype='i4',
        compressor=None, fill_value=42)
    for i in range(5):
        my_zarr[i * 10] = i

    # assign it to the asdf tree
    print("Assigning zarr array to asdf tree")
    af = asdf.AsdfFile()
    my_zarr.array_storage = 'internal'
    af['my_zarr'] = my_zarr

    # write out the asdf file
    print(f"Saving asdf file to {fn}")
    af.write_to(fn)

    # remove the zarr storage
    print(f"Removing temporary storage")
    shutil.rmtree(tmp_dir)

    # now let's open up the file (which we can read like a normal asdf file)
    print(f"Opening asdf file at {fn}")
    af = asdf.open(fn)
    print("Checking zarr data")
    a = af['my_zarr']
    for i in range(5):
        assert a[i * 10] == i
    # also check unwritten blocks were written
    for i in range(5, 10):
        assert a[i * 10] == 42
    af.close()

    print("Checking writing")
    af = asdf.open(fn, mode='rw')
    a = af['my_zarr']
    a[5] = 26
    assert a[5] == 26
    af.close()

    print("Checking write was flushed to disk")
    af = asdf.open(fn, mode='r')
    a = af['my_zarr']
    assert a[5] == 26
    af.close()

    print("all good!")

