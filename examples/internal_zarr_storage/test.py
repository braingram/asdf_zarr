from contextlib import contextmanager
import os
import sys

import asdf
import asdf_zarr
import asdf_zarr.store
import zarr


@contextmanager
def write_context(asdf_file, filename):
    with open(filename, mode='bw+') as fp:
        # convert to generic_io?
        fp = asdf.generic_io.get_file(fp, mode='rw')

        asdf_file.write_to(fp)

        # # TODO currently ignores pad_blocks
        # # write tree
        # # this will trigger to_yaml_tree for any zarrays which MUST not yet
        # # have written data
        # # at this point a valid file pointer exists and block indicies
        # # should be possible to assign
        # asdf_file._write_tree(fp)

        # # write all 'simple' blocks
        # asdf_file._write_internal_blocks_serial(fp)
        # # TODO does uri require the generic_io class?
        # asdf_file._write_external_blocks_serial(fp.uri)

        # # TODO? assign the file pointer to the chunked array storage

        # yield the file pointer
        yield fp

        # # write block index TODO make this optional?
        # asdf_file._write_block_index(fp, asdf_file)


if __name__ == '__main__':
    af = asdf.AsdfFile()

    # use the asdfile (and block manager) for storage
    store = asdf_zarr.store.InternalStore(af.blocks)

    # make a zarray using this storage (this can't yet be used for writing)
    # creation of the array will make the .zarray file but won't create chunks
    af['my_zarr'] = zarr.zeros(store=store, shape=100, chunks=10,
                               dtype='i4', compressor=None)

    # start a write context that will open a file (needs to be seekable)
    # and allow writing/reading chunks of data as blocks inside the asdf file
    with write_context(af, 'test_internal_zarr.asdf') as f:
        # now the array can be modified as a destination file is open for writing
        a = af['my_zarr']
        # write to each chunk
        for i in range(10):
            a[i * 10] = i
            assert a[i * 10] == i


    # now let's open up the file (which we can read like a normal asdf file)
    af = asdf.open('test_internal_zarr.asdf')
    a = af['my_zarr']

    # and check the data was written
    for i in range(10):
        assert a[i * 10] == i
    print("all good!")
