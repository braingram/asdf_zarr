import asdf


class ZarrConverter(asdf.extension.Converter):
    tags = ["asdf://stsci.edu/example-project/tags/zarr-*"]
    types = ["zarr.core.Array"]

    def to_yaml_tree(self, obj, tags, ctx):
        from . import util

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

        chunk_store = util.decode_storage(node['store'])
        if 'meta' in node:
            # separate meta and chunk stores
            store = util.decode_storage(node['meta_store'])
        else:
            store = chunk_store
        # TODO mode, version, path_str?
        return zarr.open(store=store, chunk_store=chunk_store)
