import asdf



class ZarrConverter(asdf.extension.Converter):
    tags = ["asdf://stsci.edu/example-project/tags/zarr-*"]
    types = ["zarr.core.Array"]

    def to_yaml_tree(self, obj, tags, ctx):
        #import pdb; pdb.set_trace()

        # defer import
        import json

        import zarr

        from . import store


        if isinstance(obj.store, zarr.storage.DirectoryStore):
            zarray = json.loads(obj.store['.zarray'])
            return {'source': f'file://{obj.chunk_store.path}', '.zarray': zarray}
        elif isinstance(obj.store, store.InternalStore):
            # need to know which blocks to use
            block_slice = obj.store.block_slice
            source = f'blocks://{block_slice[0]}:{block_slice[1]}'
            return {'.zarray': obj.store['.zarray'], 'source': source}

        raise NotImplementedError(f"zarr.store type {type(obj.store)} not supported")

    def from_yaml_tree(self, node, tag, ctx):
        #import pdb; pdb.set_trace()

        # defer import
        import copy
        import zarr

        from . import store


        # look at source to determine the zarr storage type
        source = node['source']
        source_type, source_info = source.split('://')

        if source_type == 'file':
            # if 'file://' source contains path for a DirectoryStore
            meta_store = {'.zarray': copy.deepcopy(node['.zarray'])}
            return zarr.open(store=meta_store, chunk_store=zarr.storage.DirectoryStore(source_info))
        elif source_type == 'blocks':
            # if 'blocks://' source contains indices of internal asdf blocks
            # internal store needs block manager, zarray, block indices
            zarray = copy.deepcopy(node['.zarray'])
            block_slice = [int(i) for i in source_info.split(':')]
            internal_store = store.InternalStore(ctx.block_manager,
                                                 zarray=zarray,
                                                 block_slice=block_slice)
            return zarr.open(store=internal_store)
