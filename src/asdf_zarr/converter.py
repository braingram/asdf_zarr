import json
import math

import asdf

import numpy
import zarr

from . import util
from . import storage
# TODO convert imports to local to avoid imports on extension loading


class ZarrConverter(asdf.extension.Converter):
    tags = ["asdf://stsci.edu/example-project/tags/zarr-*"]
    types = ["zarr.core.Array"]

    def to_yaml_tree(self, obj, tag, ctx):
        storage_settings = self._get_storage_settings(obj, tag, ctx)
        if storage_settings == "internal":
            if isinstance(obj, zarr.storage.NestedDirectoryStore):
                # TODO something is odd with NestedDirectoryStore where
                # it returns chunks for listdir (when they are defined)
                # However, the key generation appears to be different
                # because attempts to get those chunks it fails because
                # the chunks have to be accessed with '/' separator
                # even if the default '.' is set
                raise NotImplementedError("NestedDirectoryStore to internal not supported")
            # TODO should we enforce no zarr compression here?
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
        from . import storage

        if '.zarray' in node and 'chunk_block_map' in node:
            # this is an internally stored zarr array
            # TODO should we enforce no zarr compression here?
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
            chunk_store = storage._build_internal_store(
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
        storage_settings = self._get_storage_settings(obj, tag, ctx)
        if storage_settings != "internal":
            return []
        return self._set_internal_blocks(obj, tag, ctx)

    def _get_storage_settings(self, obj, tag, ctx):
        if obj.chunk_store is not None:
            # data is in chunk_store, metadata is in store
            meta_store = obj.store
            chunk_store = obj.chunk_store
        else:
            meta_store = obj.store
            chunk_store = obj.store

        storage_settings = ctx.get_block_storage_settings(id(obj))
        if storage_settings is None:  # guess storage
            if isinstance(
                    chunk_store, (
                        zarr.storage.KVStore,
                        zarr.storage.MemoryStore,
                        zarr.storage.TempStore,
                        storage.InternalStore,
                    )):
                storage_settings = "internal"
                ctx.set_block_storage_settings(id(obj), storage_settings)
        return storage_settings

    def _set_internal_blocks(self, obj, tag, ctx):
        # making the block chunk map here requires knowing the index of
        # each block with chunk data
        # so first, generate/find a block for each filled chunk
        # generate a block for each filled chunk and keep track of which
        # chunks have data
        blocks = []
        chunk_key_block_index_map = {}
        for chunk_key in storage._iter_chunk_keys(obj, only_initialized=True):
            # generate data callback
            key = (id(obj), chunk_key)
            data_callback = storage._generate_chunk_data_callback(obj, chunk_key)
            blk = ctx.reserve_block(key, data_callback)
            # TODO it might be nice to use the block here
            index = ctx.find_block_index(key, data_callback)
            chunk_key_block_index_map[chunk_key] = index
            blocks.append(blk)

        # now generate a callback to return the chunk_key_block_map
        blocks.append(
            ctx.reserve_block(
                id(obj),
                storage._generate_chunk_map_callback(obj, chunk_key_block_index_map)))

        return blocks
