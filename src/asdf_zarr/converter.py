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
        # TODO how to trigger ingestion? Should I force the user
        # to explicitly make an InternalStore? I think this would work.
        # it would first have no blocks but only use temp files/the source
        # store
        if isinstance(obj.chunk_store, storage.InternalStore):
            # TODO should we enforce no zarr compression here?
            # include data from this zarr array in the asdf file
            # include the meta data in the tree
            meta = json.loads(obj.store['.zarray'])
            obj_dict = {}
            obj_dict['.zarray'] = meta
            # update callbacks
            chunk_key_block_index_map = {}
            for chunk_key in storage._iter_chunk_keys(obj, only_initialized=True):
                data_callback = storage._generate_chunk_data_callback(obj, chunk_key)
                asdf_key = obj.chunk_store._asdf_keys.get(chunk_key, asdf.util.BlockKey())
                block_index = ctx.find_block_index(asdf_key, data_callback)
                chunk_key_block_index_map[chunk_key] = block_index
            asdf_key = obj.chunk_store._chunk_block_map_asdf_key
            if asdf_key is None:
                asdf_key = asdf.util.BlockKey()
            obj_dict['chunk_block_map'] = ctx.find_block_index(
                asdf_key,
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
        # if this block uses a 'InternalStore' it uses blocks
        if not isinstance(obj.chunk_store, storage.InternalStore):
            return []

        return obj.chunk_store._asdf_keys

    def _get_storage_settings(self, obj, tag, ctx):
        if obj.chunk_store is not None:
            # data is in chunk_store, metadata is in store
            meta_store = obj.store
            chunk_store = obj.chunk_store
        else:
            meta_store = obj.store
            chunk_store = obj.store

        #storage_settings = ctx.get_block_storage_settings(id(chunk_store))
        storage_settings = None
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
