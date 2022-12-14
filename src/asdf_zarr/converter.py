import asdf
import numpy


class ZarrConverter(asdf.extension.Converter):
    tags = ["asdf://stsci.edu/example-project/tags/zarr-*"]
    types = ["zarr.core.Array"]

    def to_yaml_tree(self, obj, tags, ctx):
        # At this point, we have a zarr.core.Array(obj)
        # that is stored somewhere. This could be an directory store
        # a s3 or some other FSStore, or another asdf file.
        # We need to determine where we want to save this.
        # Do we want to keep a reference to the external store
        # or ingest the data into the asdf file that is currently
        # being written?
        #
        # If the store isinstance of storage.InternalStore or
        # if store['.storage'] == 'internal' then use a
        # storage.InternalStore to ingest the datq
        #
        # In all cases, the blocks will need to, at a later point,
        # retrieve the data to be written to the file currently
        # being created. For example, write_internal_blocks_serial
        # will call block.write(fd) on all internal blocks.

        # defer import
        import json

        import numpy
        import zarr

        from . import storage

        if obj.chunk_store is not None:
            chunk_store = obj.chunk_store
        else:
            chunk_store = obj.store
        use_internal_store = (
            isinstance(chunk_store, storage.InternalStore) or
            getattr(obj, 'array_storage', '') == 'internal')

        if use_internal_store:
            # TODO verify settings are compatible
            zarray = json.loads(obj.store['.zarray'])
            if zarray.get('compressor', None) != None:
                raise NotImplementedError("zarr array compressors are not supported in asdf")
            if zarray.get('filters', None) != None:
                raise NotImplementedError("zarr array filters are not supported in asdf")

            n_bytes_per_chunk = storage.chunk_size(zarray)

            # make blocks for data
            block_indicies = []
            for chunk_key in storage.iter_chunk_keys(zarray):
                # TODO handle zarr arrays with chunk_store == InternalStore
                if isinstance(chunk_store, storage.InternalStore):
                    # blocks already exist
                    block = chunk_store._block_kv_store[chunk_key]
                    # if this is an update, they will read and write the same fd
                    # if this is a write_to, they will read from _fd and write to fd
                else:
                    block = asdf.block.Block(
                        array_storage="internal", memmap=False, lazy_load=True, cache_data=False)
                    # Block.__init__ calls update_size which will compress
                    # the data to get the final size. Instead, let's override some
                    # internal attributes that are calculated to avoid passing
                    # in data at this point (which will end up getting cached)
                    block._data_size = n_bytes_per_chunk
                    block._size = n_bytes_per_chunk
                    block._allocated = n_bytes_per_chunk

                # configure the new block

                # setting _used is not necessary as it is only used in _pre_write which
                # at this point has already been called. So any block we add here
                # will not be removed
                #block._used = True
                # However, this means that update or write_to of a file that was loaded
                # from an asdf file with an internal zarr array will have it's blocks
                # removed, then re-added here. This isn't optimal but there is no way
                # currently for a new-style extension to 'reserve_blocks' on _pre_write.
                # This does not work for write_to as:
                #   af = open(fn)
                #   af.write_to(fn2)
                #   af['my_zarr'] = 2
                # will modify fn2 NOT fn
                # setting _used on read is sub-optimal as removal of a zarr array
                # will NOT result in removal of blocks on the next write (but will on
                # subsequent writes)

                # set up the block to return the correct chunk data when write is called
                import weakref

                def wrap_write(block, store_ref, chunk_key):
                    def write(fd):
                        # fetch data from store
                        with store_ref() as store:
                            if chunk_key in store:
                                data = store[chunk_key]
                            else:
                                # TODO handle fill_value here
                                # make a filler array
                                data = storage.make_filled_chunk(zarray)
                        block._write_data(fd, numpy.frombuffer(data, dtype='uint8'))
                    block.write = write

                wrap_write(block, weakref.ref(chunk_store), chunk_key)

                if not isinstance(chunk_store, storage.InternalStore):
                    bi = ctx.block_manager.add(block)
                else:
                    # look up the block index for the existing block
                    bi = ctx.block_manager._internal_blocks.index(block)
                block_indicies.append(bi)

            source = f'blocks://{block_indicies[0]}:{block_indicies[-1]+1}'
            # setup callbacks to retrieve data when blocks are written
            return {'.zarray': zarray, 'source': source}
        else:
            # handle other storage types
            if isinstance(chunk_store, zarr.storage.DirectoryStore):
                # For a DirectoryStore use a file://<filename> source
                zarray = json.loads(obj.store['.zarray'])
                return {'source': f'file://{obj.chunk_store.path}', '.zarray': zarray}
            elif isinstance(chunk_store, zarr.storage.FSStore):
                # TODO at the moment only s3 is supported
                source = f's3://{chunk_store.path}'

                # read .zarray from store
                zarray = json.loads(obj.store['.zarray'])

                # TODO other options (client_kwargs, etc)
                client_kwargs = chunk_store.fs.client_kwargs
                return {'source': source, '.zarray': zarray, 'client_kwargs': client_kwargs}
            raise NotImplementedError(f"{self.__class__}: zarr.store type {type(chunk_store)} not supported")

    def from_yaml_tree(self, node, tag, ctx):
        # defer import
        import copy
        import warnings

        import zarr

        from . import storage


        # look at source to determine the zarr storage type
        source = node['source']
        source_type, source_info = source.split('://')

        if source_type == 'file':
            # if 'file://' source contains path for a DirectoryStore
            meta_store = {'.zarray': copy.deepcopy(node['.zarray'])}
            return zarr.open(store=meta_store, chunk_store=zarr.storage.DirectoryStore(source_info))
        elif source_type == 's3':
            # use fsspec to load an s3 store
            # TODO other options like separator, etc
            meta_store = {'.zarray': copy.deepcopy(node['.zarray'])}
            kwargs = {}
            if 'client_kwargs' in node:
                kwargs['client_kwargs'] = node['client_kwargs']
            return zarr.open(store=meta_store, chunk_store=zarr.storage.FSStore(source, **kwargs))
        elif source_type == 'blocks':
            # if 'blocks://' source contains indices of internal asdf blocks
            # internal store needs block manager, zarray, block indices
            zarray = copy.deepcopy(node['.zarray'])
            block_slice = [int(i) for i in source_info.split(':')]
            blocks = ctx.block_manager._internal_blocks[block_slice[0]:block_slice[1]]

            # setup blocks
            for block in blocks:
                if not block._lazy_load:
                    warnings.warn(
                        f"{self._class__} found a block with lazy_load==False " +
                        "this can result in reading the entire zarr array into memory "
                        "and should be avoided")

                block._lazy_load = True
                block._should_memmap = False
                block._cache_data = False
                # There is currently no way to hook this extension to
                # 'reserve_blocks' during _find_used_blocks. So to prevent blocks
                # from being removed and re-added during write we mark them
                # as used on read.
                # This does mean that blocks will not be removed if the zarr array
                # is removed and the file saved (they will however be removed if that
                # saved file is re-read and re-saved).
                block._used = True

            internal_store = storage.InternalStore(zarray, blocks)
            return zarr.open(store=internal_store)
        raise NotImplementedError(f"{self.__class__}: source {source} not supported")
