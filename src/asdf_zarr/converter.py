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
            # TODO should we enforce no zarr compression here?
            # include data from this zarr array in the asdf file
            # include the meta data in the tree
            meta = json.loads(obj.store['.zarray'])
            obj_dict = {}
            obj_dict['.zarray'] = meta
            # update callbacks
            chunk_key_block_index_map = {}
            for chunk_key in storage._iter_chunk_keys(obj, only_initialized=True):
                key = (id(obj.chunk_store), chunk_key)
                data_callback = storage._generate_chunk_data_callback(obj, chunk_key)
                block_index = ctx.find_block_index(key, data_callback)
                ctx.claim_block(block_index, key)
                chunk_key_block_index_map[chunk_key] = block_index
            obj_dict['chunk_block_map'] = ctx.find_block_index(
                id(obj.chunk_store),
                storage._generate_chunk_map_callback(obj, chunk_key_block_index_map))
            ctx.claim_block(obj_dict['chunk_block_map'], id(obj))
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

            #    zarray_meta,
            #    chunk_block_map,
            #    ctx,
            #    node['.zarray'].get('dimension_separator', '.'),
            #)
            # # now that we have an object, claim the blocks
            # # use id(chunk_store) in case this is used for multiple zarr arrays
            # ctx.claim_block(chunk_block_map_index, id(chunk_store))
            # for chunk_key in storage._iter_chunk_keys(obj, only_initialized=True):
            #     block_index = chunk_store._key_to_block_index(chunk_key)
            #     ctx.claim_block(block_index, (id(chunk_store), chunk_key))

            # TODO read/write mode here
            obj = zarr.open_array(store=store, chunk_store=chunk_store)
            return obj

        chunk_store = util.decode_storage(node['store'])
        if 'meta' in node:
            # separate meta and chunk stores
            store = util.decode_storage(node['meta_store'])
        else:
            store = chunk_store
        # TODO mode, version, path_str?
        obj = zarr.open(store=store, chunk_store=chunk_store)
        return obj

    def reserve_blocks(self, obj, tag, ctx):
        storage_settings = self._get_storage_settings(obj, tag, ctx)

        if storage_settings != "internal":
            return []

        # if we're using internal storage, return keys for
        # the chunk_block_map
        keys = [id(obj), ]
        # all initialized chunks
        for chunk_key in storage._iter_chunk_keys(obj, only_initialized=True):
            keys.append((id(obj.chunk_store), chunk_key))
        return keys

    def _get_storage_settings(self, obj, tag, ctx):
        if obj.chunk_store is not None:
            # data is in chunk_store, metadata is in store
            meta_store = obj.store
            chunk_store = obj.chunk_store
        else:
            meta_store = obj.store
            chunk_store = obj.store

        storage_settings = ctx.get_block_storage_settings(id(chunk_store))
        if storage_settings is None:  # guess storage
            if isinstance(
                    chunk_store, (
                        zarr.storage.KVStore,
                        zarr.storage.MemoryStore,
                        zarr.storage.TempStore,
                        storage.InternalStore,
                    )):
                storage_settings = "internal"
                ctx.set_block_storage_settings(id(chunk_store), storage_settings)
        return storage_settings
