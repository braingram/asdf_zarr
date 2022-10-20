This is an experimental asdf extension to allow writing and reading
of zarr arrays where chunks are stored as separate blocks inside a
single asdf file.

The extension currently requires access to the block_manager during
asdf file reading so is only compatible with a slightly modified asdf
found [here](https://github.com/braingram/asdf/tree/feature/zarr_chunking).

Example uses can be found in [examples](examples) including:
- external (DirectoryStore) zarr chunk storage: [external_zarr_storage](examples/external_zarr_storage/test.py)
- internal (using blocks) zarr chunk storage: [internal_zarr_storage](examples/internal_zarr_storage/test.py)

The external storage example can be run with an unmodified asdf (however
support for internal storage would need to be removed to allow this
extension to work).
