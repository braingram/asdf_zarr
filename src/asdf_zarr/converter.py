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
        storage_settings = ctx.get_array_storage(obj)
        breakpoint()

        if storage_settings == "internal":
            # TODO should we enforce no zarr compression here?
            # include data from this zarr array in the asdf file
            # include the meta data in the tree
            meta = json.loads(obj.store['.zarray'])
            obj_dict = {}
            obj_dict['.zarray'] = meta
            chunk_store = obj.chunk_store or obj.store
            # update callbacks
            chunk_key_block_index_map = {}
            for chunk_key in storage._iter_chunk_keys(obj, only_initialized=True):
                data_callback = storage._generate_chunk_data_callback(obj, chunk_key)
                if hasattr(chunk_store, '_chunk_asdf_key'):
                    asdf_key = chunk_store._chunk_asdf_key[chunk_key]
                else:
                    asdf_key = asdf.util.BlockKey(obj)
                block_index = ctx.find_block_index(asdf_key, data_callback)
                chunk_key_block_index_map[chunk_key] = block_index
            obj_dict['chunk_block_map'] = ctx.find_block_index(
                chunk_store,
                storage._generate_chunk_map_callback(obj, chunk_key_block_index_map))
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
            chunk_block_map_index = node['chunk_block_map']

            chunk_store = storage.InternalStore(ctx, chunk_block_map_index, zarray_meta)

            # TODO read/write mode here
            obj = zarr.open_array(store=store, chunk_store=chunk_store)
            # now that we have an object, assign the block keys
            chunk_store._assign_block_keys(obj, ctx)
            ctx.set_array_storage(obj, "internal")
            return obj

        chunk_store = util.decode_storage(node['store'])
        if 'meta' in node:
            # separate meta and chunk stores
            store = util.decode_storage(node['meta_store'])
        else:
            store = chunk_store
        # TODO mode, version, path_str?
        obj = zarr.open(store=store, chunk_store=chunk_store)
        ctx.set_array_storage(obj, "external")
        return obj

    def _get_storage_settings(self, obj, tag, ctx):
        if obj.chunk_store is not None:
            # data is in chunk_store, metadata is in store
            meta_store = obj.store
            chunk_store = obj.chunk_store
        else:
            meta_store = obj.store
            chunk_store = obj.store

        # ignore compression for now
        # currently, this creates a block for the chunk_store, that's ok we'll
        # use that for the map
        array_storage = ctx.get_array_storage(obj)
        # using these options the default is now internal
        #storage_settings = None
        #if storage_settings is None:  # guess storage
        #    if isinstance(
        #            chunk_store, (
        #                zarr.storage.KVStore,
        #                zarr.storage.MemoryStore,
        #                zarr.storage.TempStore,
        #                storage.InternalStore,
        #            )):
        #        storage_settings = "internal"
        #        ctx.set_block_storage_settings(_get_obj_key(chunk_store), storage_settings)
        return array_storage
