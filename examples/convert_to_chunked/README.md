Example that converts an asdf file with large arrays
to a asdf file with chunked arrays.

Tested with JWST montage of M 16
jw02739-o001_t001_nircam_f444w-f470n_i2d.fits

The script drops all non-converted tags as some bounding
box in the meta data was failing validation on write.
