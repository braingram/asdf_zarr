import itertools
import json
import math

import asdf
import numpy
import zarr


def make_filled_chunk(zarray):
    arr = numpy.empty(zarray['chunks'], zarray['dtype'])
    arr[:] = zarray['fill_value']
    return arr


def chunk_size(zarray):
    """Compute the size, in bytes, of a single chunk"""
    dt = numpy.dtype(zarray['dtype'])
    n = dt.itemsize
    for c in zarray['chunks']:
        n *= c
    return n


def iter_chunk_keys(zarray):
    """Using zarray metadata iterate over chunk keys"""
    dimension_separator = zarray.get('dimension_separator', '.')

    # make blocks and map them to the internal kv store
    # compute number of chunks (across all axes)
    chunk_counts = [math.ceil(s / c) for (s, c) in
                    zip(zarray['shape'], zarray['chunks'])]

    # iterate over all chunk keys
    chunk_iter = itertools.product(*[range(c) for c in chunk_counts])
    for c in chunk_iter:
        key = dimension_separator.join([str(i) for i in c])
        yield key


class InternalStore(zarr.storage.KVStore):
    """
    A key-value store for mapping chunks to blocks
    and for storing metadata (.zarray) in the tree

    Each zarray should have it's own store (so no additional keys are needed)
    """
    def __init__(self, zarray, blocks):
        self._zarray = zarray

        # look up blocks
        self._blocks = blocks

        # generate a key
        # value store of blocks that can be looked up based on the
        # chunk key from zarr
        self._block_kv_store = {}

        # TODO check for no compression and other unsupported features
        if self._zarray.get('compressor', None) is not None:
            raise NotImplementedError(
                f"{self.__class__} does not support compressor")
        if self._zarray.get('filters', None) is not None:
            raise NotImplementedError(
                f"{self.__class__} does not support filters")

        for (chunk_key, block) in zip(iter_chunk_keys(self._zarray), self._blocks):
            self._block_kv_store[chunk_key] = block

    def _get_chunk(self, key):
        """Given a zarr chunk key, look up and read the block"""
        print(f"_get_chunk {key=}")
        # look up the block
        block = self._block_kv_store[key]
        # read block
        return block.read_data()

    def _set_chunk(self, key, value):
        """Given a zarr chunk key, look up and write to the block"""
        print(f"_set_chunk {key=}")
        block = self._block_kv_store[key]
        block.rewrite_data(numpy.frombuffer(value, dtype='uint8'))

    def __getitem__(self, key):
        if key == '.zgroup':
            raise KeyError(".zgroup not found")
        if key == '.zarray':
            if self._zarray is None:
                raise KeyError(".zarray does not exist")
            else:
                return self._zarray
        return self._get_chunk(key)

    def __setitem__(self, key, value):
        if key == '.zaarray':
            self._set_zarray(json.loads(value.decode()))
            return
        self._set_chunk(key, value)

    def __delitem__(self, key):
        if key == '.zarray':
            self._zarray = None
            return
        raise NotImplementedError("cannot delete chunks")
