import asdf



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

        use_internal_store = (
            isinstance(obj.store, storage.InternalStore) or
            getattr(obj, 'array_storage', '') == 'internal')

        if use_internal_store:
            # TODO verify settings are compatible
            zarray = json.loads(obj.store['.zarray'])

            n_bytes_per_chunk = storage.chunk_size(zarray)

            # make blocks for data
            block_indicies = []
            for chunk_key in storage.iter_chunk_keys(zarray):
                block = asdf.block.Block(
                    array_storage="internal", memmap=False, lazy_load=True, cache_data=False)

                # configure the new block
                block.output_compression = None
                block._used = True
                block._data_size = block._size = n_bytes_per_chunk

                # set up the block to return the correct chunk data when write is called
                import weakref

                def wrap_write(block, store_ref, chunk_key):
                    def write(fd):
                        # fetch data from store
                        with store_ref() as store:
                            #print(f"fetching data for chunk {chunk_key}")
                            # TODO handle fill_value here
                            if chunk_key in store:
                                data = store[chunk_key]
                            else:
                                raise Exception("Fill values are not yet supported")
                        block.write_data(fd, numpy.frombuffer(data, dtype='uint8'))
                    block.write = write

                wrap_write(block, weakref.ref(obj.store), chunk_key)

                bi = ctx.block_manager.add(block)
                block_indicies.append(bi)

            source = f'blocks://{block_indicies[0]}:{block_indicies[-1]+1}'
            # setup callbacks to retrieve data when blocks are written
            return {'.zarray': zarray, 'source': source}
        else:
            # handle other storage types
            if isinstance(obj.store, zarr.storage.DirectoryStore):
                # For a DirectoryStore use a file://<filename> source
                zarray = json.loads(obj.store['.zarray'])
                return {'source': f'file://{obj.chunk_store.path}', '.zarray': zarray}
            elif isinstance(obj.store, zarr.storage.FSStore):
                # TODO at the moment only s3 is supported
                source = f's3://{obj.store.path}'

                # read .zarray from store
                zarray = json.loads(obj.store['.zarray'])

                # TODO other options (client_kwargs, etc)
                client_kwargs = obj.store.fs.client_kwargs
                return {'source': source, '.zarray': zarray, 'client_kwargs': client_kwargs}
            raise NotImplementedError(f"{self.__class__}: zarr.store type {type(obj.store)} not supported")

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

            internal_store = storage.InternalStore(zarray, blocks)
            return zarr.open(store=internal_store)
        raise NotImplementedError(f"{self.__class__}: source {source} not supported")
