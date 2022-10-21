Installing moto and s3fs required installing specific versions
of several libraries to allow pip to avoid dependency issues

I believe the following worked for me. Install these in this order:

- boto3==1.24.59
- botocore==1.27.59
- aiobotocore
- s3fs
- moto
- flask
- flask_cors

This should allow you to call [run_server.sh](run_server.sh)
which will start a mock s3 server on localhost port 5555.

While the mock server is running, run test.py to test the
s3 zarr array asdf integration.
