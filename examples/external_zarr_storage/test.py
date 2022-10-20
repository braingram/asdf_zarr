import os

import asdf
import zarr


af = asdf.AsdfFile()
store = zarr.storage.DirectoryStore('zarr_data')
if os.path.exists('zarr_data'):
    a = zarr.open(store=store)
else:
    a = zarr.zeros(store=store, shape=(1000, 1000), chunks=(100, 100), dtype='i4')
a[42, 26] = 42

af['my_zarr'] = a
af.write_to('test_zarr.asdf')

af = asdf.open('test_zarr.asdf')
a = af['my_zarr']
assert a[42, 26] == 42
print("all good!")
