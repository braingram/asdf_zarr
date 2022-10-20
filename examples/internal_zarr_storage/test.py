from contextlib import contextmanager
import os
import sys

import asdf
import asdf_zarr
import asdf_zarr.store
import zarr


@contextmanager
def write_context(asdf_file, filename, *args, **kwargs):
    """
    Use a context manager to allow the file to stay open so chunks
    can be written to the blocks inside the asdf file
    """
    with open(filename, mode='bw+') as fp:
        # convert to generic_io
        # TODO map modes, only certain ones (possibly rw) are supported
        fp = asdf.generic_io.get_file(fp, mode='rw')

        asdf_file.write_to(fp, *args, **kwargs)

        # yield the file pointer
        yield fp

        # TODO write all unwritten chunks with fill values

        # TODO release file pointer from blocks?


if __name__ == '__main__':
    af = asdf.AsdfFile()

    # use the asdfile (and block manager) for storage
    store = asdf_zarr.store.InternalStore(af.blocks)

    # make a zarray using this storage (this can't yet be used for writing)
    # creation of the array will make the .zarray file but won't create chunks
    af['my_zarr'] = zarr.create(100, store=store, chunks=10, dtype='i4', compressor=None, fill_value=42)

    # start a write context that will open a file (needs to be seekable)
    # and allow writing/reading chunks of data as blocks inside the asdf file
    with write_context(af, 'test_internal_zarr.asdf') as f:
        # now the array can be modified as a destination file is open for writing
        a = af['my_zarr']
        # write to each chunk
        for i in range(5):
            a[i * 10] = i
            assert a[i * 10] == i
        store.write_remaining_blocks()


    # now let's open up the file (which we can read like a normal asdf file)
    af = asdf.open('test_internal_zarr.asdf')
    a = af['my_zarr']

    # and check the data was written
    for i in range(5):
        assert a[i * 10] == i
    # also check unwritten blocks were written
    for i in range(5, 10):
        assert a[i * 10] == 42
    print("all good!")
