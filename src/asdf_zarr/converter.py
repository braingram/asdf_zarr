import json
import math

import asdf

import numpy
import zarr


MISSING_CHUNK = -1


def _iter_chunk_keys(zarray, only_initialized=False):
    """Using zarray metadata iterate over chunk keys"""
    if only_initialized:
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
        return numpy.frombuffer(zarray.store.get(chunk_key), dtype='uint8')
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


def _build_internal_store(zarray_meta, chunk_block_map, ctx, sep):
    # go from key to chunk coordinate
    # use coordinate and chunk_block_map to lookup block using ctx
    class InternalStore(zarr.storage.Store):
        def __init__(self, chunk_block_map, ctx, sep):
            super().__init__()
            self._ctx = ctx
            self._chunk_block_map = chunk_block_map
            self.__sep = sep

        def _sep(self, key):
            if self.__sep is None:
                return key
            return key.split(self.__sep)

        def _coords(self, key):
            return tuple([int(sk) for sk in self._sep(key)])

        def __getitem__(self, key):
            coords = self._coords(key)
            index = int(self._chunk_block_map[coords])
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

    return InternalStore(chunk_block_map, ctx, sep)


class ZarrConverter(asdf.extension.Converter):
    tags = ["asdf://stsci.edu/example-project/tags/zarr-*"]
    types = ["zarr.core.Array"]

    def to_yaml_tree(self, obj, tag, ctx):
        from . import util

        storage = ctx.get_block_storage_settings(id(obj))
        if storage == "internal":
            if isinstance(obj, zarr.storage.NestedDirectoryStore):
                # TODO something is odd with NestedDirectoryStore where
                # it returns chunks for listdir (when they are defined)
                # However, the key generation appears to be different
                # because attempts to get those chunks it fails because
                # the chunks have to be accessed with '/' separator
                # even if the default '.' is set
                raise NotImplementedError("NestedDirectoryStore to internal not supported")
            # include data from this zarr array in the asdf file
            # include the meta data in the tree
            meta = json.loads(obj.store['.zarray'])
            obj_dict = {}
            obj_dict['.zarray'] = meta
            # update callbacks
            self._set_internal_blocks(obj, tag, ctx)
            obj_dict['chunk_block_map'] = ctx.find_block_index(id(obj))
            return obj_dict

        if obj.chunk_store is not None:
            # data is in chunk_store, metadata is in store
            meta_store = obj.store
            chunk_store = obj.chunk_store
        else:
            meta_store = obj.store
            chunk_store = obj.store

        obj_dict = {}
        if meta_store is not chunk_store:
            # encode meta store
            obj_dict['meta_store'] = util.encode_storage(meta_store)
        obj_dict['store'] = util.encode_storage(chunk_store)
        # TODO mode, version, path_str?
        return obj_dict


    def from_yaml_tree(self, node, tag, ctx):
        import zarr

        from . import util

        if '.zarray' in node and 'chunk_block_map' in node:
            # this is an internally stored zarr array
            # load the meta data into memory
            store = zarr.storage.KVStore({'.zarray': json.dumps(node['.zarray'])})
            # setup an InternalStore to read block data (when requested)
            zarray_meta = node['.zarray']
            cdata_shape = tuple(math.ceil(s / c)
                         for s, c in zip(zarray_meta['shape'], zarray_meta['chunks']))
            chunk_block_map = numpy.frombuffer(
                ctx.load_block(node['chunk_block_map']),
                dtype='int32').reshape(cdata_shape)
            # TODO clean up these arguments
            chunk_store = _build_internal_store(
                zarray_meta,
                chunk_block_map,
                ctx,
                node['.zarray'].get('dimension_separator', '.'))
            # TODO read/write mode here
            return zarr.open_array(store=store, chunk_store=chunk_store)

        chunk_store = util.decode_storage(node['store'])
        if 'meta' in node:
            # separate meta and chunk stores
            store = util.decode_storage(node['meta_store'])
        else:
            store = chunk_store
        # TODO mode, version, path_str?
        return zarr.open(store=store, chunk_store=chunk_store)

    def reserve_blocks(self, obj, tag, ctx):
        #return [ctx.reserve_block(id(obj), lambda: np.ndarray(len(obj.payload), dtype="uint8", buffer=obj.payload))]
        storage = ctx.get_block_storage_settings(id(obj))
        if storage != "internal":
            return []
        return self._set_internal_blocks(obj, tag, ctx)

    def _set_internal_blocks(self, obj, tag, ctx):
        # making the block chunk map here requires knowing the index of
        # each block with chunk data
        # so first, generate/find a block for each filled chunk
        # generate a block for each filled chunk and keep track of which
        # chunks have data
        blocks = []
        chunk_key_block_index_map = {}
        for chunk_key in _iter_chunk_keys(obj, only_initialized=True):
            # generate data callback
            key = (id(obj), chunk_key)
            data_callback = _generate_chunk_data_callback(obj, chunk_key)
            blk = ctx.reserve_block(key, data_callback)
            # TODO it might be nice to use the block here
            index = ctx.find_block_index(key, data_callback)
            chunk_key_block_index_map[chunk_key] = index
            blocks.append(blk)

        # now generate a callback to return the chunk_key_block_map
        blocks.append(
            ctx.reserve_block(
                id(obj),
                _generate_chunk_map_callback(obj, chunk_key_block_index_map)))

        return blocks
