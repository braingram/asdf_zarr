import os

import asdf
import boto3
import zarr


# endpoint used to fake s3
endpoint_url = 'http://127.0.0.1:5555'
bucket = 'test_bucket'
url = f's3://{bucket}/my_zarr'

# connect to s3
conn = boto3.resource('s3', endpoint_url=endpoint_url)

# make a new bucket to store the data
bucket = conn.create_bucket(Bucket=bucket)

# clear bucket so zarr.zeros can create an array
bucket.objects.delete()

# create a fsstore using the s3 url
store = zarr.storage.FSStore(url, client_kwargs={'endpoint_url': endpoint_url})

# create a new chunked array
a = zarr.zeros(store=store, shape=(1000, 1000), chunks=(100, 100), dtype='i4')
# write to the array
a[42, 26] = 42

# create an asdf file and save the chunked array
af = asdf.AsdfFile()
af['my_zarr'] = a
af.write_to('test_zarr.asdf')

# open the asdf file and check the chunked array loaded
af = asdf.open('test_zarr.asdf')
a = af['my_zarr']
assert a[42, 26] == 42
print("all good!")
