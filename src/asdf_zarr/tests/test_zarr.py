import asdf
import asdf_zarr
import zarr


def test_directory_store(tmp_path):
    af = asdf.AsdfFile()
    data_dir = tmp_path / 'zarr_data'
    store = zarr.storage.DirectoryStore(data_dir)
    if data_dir.exists():
        a = zarr.open(store=store)
    else:
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
    af = asdf.AsdfFile()

    store = asdf_zarr.store.InternalStore(af.blocks)
    af['my_zarr'] = zarr.create(100, store=store, chunks=10, dtype='i4', compressor=None, fill_value=42)

    fn = tmp_path / 'test.asdf'
    raw_fp = fn.open(mode='bw+')
    fp = asdf.generic_io.get_file(raw_fp, mode='rw')
    af.write_to(fp)
    a = af['my_zarr']
    for i in range(5):
        a[i * 10] = i
        assert a[i * 10] == i
    store.write_remaining_blocks()
    raw_fp.close()

    with asdf.open(fn) as af:
        a = af['my_zarr']
        assert isinstance(a, zarr.core.Array)
        assert isinstance(a.chunk_store, asdf_zarr.store.InternalStore)
        for i in range(5):
            assert a[i * 10] == i
        for i in range(5, 10):
            assert a[i * 10] == 42


def test_convert_chunked(tmp_path):
    pass
