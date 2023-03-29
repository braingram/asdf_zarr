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
    def __init__(self, ctx, chunk_block_map_index, zarray_meta):
        super().__init__()


        self._ctx = ctx
        self._sep = zarray_meta.get('dimension_separator', '.')

        # the chunk_block_map contains block indicies
        # organized in an array shaped like the chunks
        # so for a zarray with 4 x 5 chunks (dimension 1
        # split into 4 chunks) the chunk_block_map will be
        # 4 x 5
        cdata_shape = tuple(math.ceil(s / c)
                     for s, c in zip(zarray_meta['shape'], zarray_meta['chunks']))
        self._chunk_block_map = numpy.frombuffer(
            ctx.load_block(chunk_block_map_index, by_index=True), dtype='int32').reshape(cdata_shape)

        # claim the bock used for the map
        self._ctx.claim_block(chunk_block_map_index, id(self))

        # reorganize the map into a set and claim the block indices
        self._chunk_block_map_keys = set()
        for coord in numpy.transpose(numpy.nonzero(self._chunk_block_map != MISSING_CHUNK)):
            coord = tuple(coord)
            block_index = int(self._chunk_block_map[coord])
            chunk_key = self._sep.join((str(c) for c in tuple(coord)))
            self._ctx.claim_block(block_index, (id(self), chunk_key))
            self._chunk_block_map_keys.add(chunk_key)

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

    def __getitem__(self, key):
        if key not in self._chunk_block_map_keys:
            return None
        return self._ctx.load_block((id(self), key))

    def __setitem__(self, key, value):
        raise NotImplementedError("writing to InternalStore not yet supported")

    def __delitem__(self, key):
        raise NotImplementedError("deleting chunks in InternalStore not yet supported")

    def __iter__(self):
        yield from self._chunk_block_map_keys

    def __len__(self):
        return len(self._chunk_block_map_keys)

    def listdir(self, path):
        # allows efficient zarr.storage.listdir
        if path:
            raise NotImplementedError("path argument not supported by InternalStore.listdir")
        return list(self)
