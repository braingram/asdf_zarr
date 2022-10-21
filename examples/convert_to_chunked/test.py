from contextlib import contextmanager
import os
import sys

import asdf
import asdf_zarr
import asdf_zarr.store
import numpy
import zarr


input_filename = 'input.fits'
arrays_to_chunk = ['con', 'data']
chunks = [1000, 1000]


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
    # open up the input asdf file
    input_af = asdf.open(input_filename)

    # open up a file to write to
    output_af = asdf.AsdfFile()

    # first make the tree
    for k in input_af.keys():
        if k in arrays_to_chunk:
            # set up a store but don't write the data
            store = asdf_zarr.store.InternalStore(output_af.blocks)
            src = input_af[k]
            output_af[k] = zarr.create(
                src.shape, dtype=src.dtype, store=store, chunks=chunks,
                compressor=None)
        else:
            #output_af[k] = input_af[k]
            pass  # ignore other keys for now

    with write_context(output_af, 'chunked.asdf') as f:
        for k in arrays_to_chunk:
            # now copy over array contents
            output_af[k][:] = input_af[k]
            output_af[k].store.write_remaining_blocks()

    read_af = asdf.open('chunked.asdf')
    for k in arrays_to_chunk:
        assert numpy.all(numpy.isclose(input_af[k], read_af[k]))

    print("all good!")
