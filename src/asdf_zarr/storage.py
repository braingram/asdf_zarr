import json
import math

import numpy
import zarr


MISSING_CHUNK = -1


def _iter_chunk_keys(zarray, only_initialized=False):
    """Using zarray metadata iterate over chunk keys"""
    if only_initialized:
        # TODO this does not work for NestedDirectoryStore
        #if isinstance(zarray.chunk_store, zarr.storage.NestedDirectoryStore):
        #    raise NotImplementedError("zarr.storage.NestedDirectoryStore is not supported")
        for k in zarr.storage.listdir(zarray.chunk_store):
            if k == '.zarray':
                continue
            yield k
        return
    # load meta
    zarray_meta = json.loads(zarray.store['.zarray'])
    dimension_separator = zarray_meta.get('dimension_separator', '.')

    # make blocks and map them to the internal kv store
    # compute number of chunks (across all axes)
    chunk_counts = [math.ceil(s / c) for (s, c) in
                    zip(zarray_meta['shape'], zarray_meta['chunks'])]

    # iterate over all chunk keys
    chunk_iter = itertools.product(*[range(c) for c in chunk_counts])
    for c in chunk_iter:
        key = dimension_separator.join([str(i) for i in c])
        yield key


def _generate_chunk_data_callback(zarray, chunk_key):
    def chunk_data_callback(zarray=zarray, chunk_key=chunk_key):
        return numpy.frombuffer(zarray.chunk_store.get(chunk_key), dtype='uint8')
    return chunk_data_callback


def _generate_chunk_map_callback(zarray, chunk_key_block_index_map):
    # make an array
    def chunk_map_callback(zarray=zarray, chunk_key_block_index_map=chunk_key_block_index_map):
        chunk_map = numpy.zeros(zarray.cdata_shape, dtype='int32')
        chunk_map[:] = MISSING_CHUNK  # set all as uninitialized
        zarray_meta = json.loads(zarray.store['.zarray'])
        dimension_separator = zarray_meta.get('dimension_separator', '.')
        for k in _iter_chunk_keys(zarray, only_initialized=True):
            index = chunk_key_block_index_map[k]
            coords = tuple([int(sk) for sk in k.split(dimension_separator)])
            chunk_map[coords] = index
        return chunk_map
    return chunk_map_callback


class InternalStore(zarr.storage.Store):
    def __init__(self, chunk_block_map, ctx, sep):
        super().__init__()
        self._ctx = ctx
        self._chunk_block_map = chunk_block_map
        self._sep = sep
        # TODO for updates to zarr
        # arrays that require addition of new chunks (or perhaps even
        # overwriting existing chunks) a system for adding blocks
        # on top of those already reserved is needed.

    def _sep_key(self, key):
        if self._sep is None:
            return key
        return key.split(self._sep)

    def _coords(self, key):
        return tuple([int(sk) for sk in self._sep_key(key)])

    def _key_to_block_index(self, key):
        coords = self._coords(key)
        return int(self._chunk_block_map[coords])

    def __getitem__(self, key):
        index = self._key_to_block_index(key)
        if index == MISSING_CHUNK:
            return None
        return self._ctx.load_block(index)

    def __setitem__(self, key, value):
        raise NotImplementedError("writing to InternalStore not yet supported")

    def __delitem__(self, key):
        raise NotImplementedError("deleting chunks in InternalStore not yet supported")

    def __iter__(self):
        for coord in numpy.transpose(numpy.nonzero(self._chunk_block_map != MISSING_CHUNK)):
            coord = tuple(coord)
            yield self._sep.join((str(c) for c in coord))

    def __len__(self):
        return numpy.count_nonzero(self._chunk_block_map != MISSING_CHUNK)

    def listdir(self, path):
        # allows efficient zarr.storage.listdir
        if path:
            raise NotImplementedError("path argument not supported by InternalStore.listdir")
        return list(self)


def _build_internal_store(zarray_meta, chunk_block_map, ctx, sep):
    return InternalStore(chunk_block_map, ctx, sep)
