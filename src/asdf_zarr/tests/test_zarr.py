import asdf
import zarr

import asdf_zarr


def test_directory_store(tmp_path):
    """
    Save a reference to a DirectoryStore backed zarr array.
    This should, when loaded, reuse the DirectoryStore and not
    include the data in the asdf file.
    """
    af = asdf.AsdfFile()
    data_dir = tmp_path / 'zarr_data'
    store = zarr.storage.DirectoryStore(data_dir)
    a = zarr.zeros(store=store, shape=(1000, 1000), chunks=(100, 100), dtype='i4')
    a[42, 26] = 42
    af['my_zarr'] = a
    fn = tmp_path / 'test_zarr.asdf'
    af.write_to(fn)

    with asdf.open(fn) as af:
        a = af['my_zarr']
        assert isinstance(a, zarr.core.Array)
        assert isinstance(a.chunk_store, zarr.storage.DirectoryStore)
        assert a[42, 26] == 42
        assert a[0, 0] == 0


def test_internal(tmp_path):
    """
    valid metadata keys are:
        - zarr_format = 2 for zarr format version 2
        - shape = array shape
        - chunks = chunk sizes
        - dtype = endianess, type, length = "<i4"
        - compressor = None/null if no compression
        - fill_value = None/null if no fill value
        - order = 'C' or 'F'
        - filters = None
        - (optional) dimension_separator = '.' '/' or missing
    various zarr configuration options like:
        - path separator
        - dtype
        - dimensions
        - chunk shapes
        - memory order: C or F (order within a chunk)
        - partial chunks
        - auto-sized chunks? chunks=True, chunks=False, chunks=(..., None...)
    Check for errors as well:
        - no compression
        - no filters
        - no object_codec?
    """
    af = asdf.AsdfFile()
    data_dir = tmp_path / 'zarr_data'
    store = zarr.storage.DirectoryStore(data_dir)
    a = zarr.zeros(
        store=store, shape=(1000, 1000), chunks=(100, 100), dtype='i4',
        compressor=False)
    a[42, 26] = 42
    a.array_storage = "internal"
    af['my_zarr'] = a
    fn = tmp_path / 'test_zarr.asdf'
    af.write_to(fn)

    with asdf.open(fn) as af:
        a = af['my_zarr']
        assert isinstance(a, zarr.core.Array)
        assert isinstance(a.chunk_store, asdf_zarr.storage.InternalStore)
        assert a[42, 26] == 42
        assert a[0, 0] == 0


def test_internal_rw(tmp_path):
    af = asdf.AsdfFile()
    data_dir = tmp_path / 'zarr_data'
    store = zarr.storage.DirectoryStore(data_dir)
    a = zarr.zeros(
        store=store, shape=(1000, 1000), chunks=(100, 100), dtype='i4',
        compressor=False)
    a[42, 26] = 42
    a.array_storage = "internal"
    af['my_zarr'] = a
    fn = tmp_path / 'test_zarr.asdf'
    af.write_to(fn)

    with asdf.open(fn, mode="rw") as af:
        a = af['my_zarr']
        assert isinstance(a, zarr.core.Array)
        assert isinstance(a.chunk_store, asdf_zarr.storage.InternalStore)
        assert a[42, 26] == 42
        assert a[0, 0] == 0
        a[42, 26] = 26
        assert a[42, 26] == 26

    with asdf.open(fn, mode="r") as af:
        a = af['my_zarr']
        assert a[42, 26] == 26
        assert a[0, 0] == 0


def test_convert_chunked(tmp_path):
    pass
