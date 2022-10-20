import itertools
import json
import math

import asdf
import numpy
import zarr


def block_to_zarr_chunk_block(block):
    zarr_chunk_block = ZarrChunkBlock(block._data_size)
    zarr_chunk_block._written = True
    zarr_chunk_block._fp = block._fd
    zarr_chunk_block.offset = block.offset
    return zarr_chunk_block


class ZarrChunkBlock(asdf.block.Block):
    """
    """
    def __init__(self, data_size):
        # if data is None, size is 0, data size is 0
        super().__init__(data=None)
        self._data_size = data_size
        self._size = data_size
        self._used = True

        # on creation, there is no associated file pointer
        # this will be set on write (called during write_to)
        self._fp = None
        # was this block written at least once?
        self._written = False

    def write(self, fd):
        # capture the file pointer for later use reading
        # and writing chunks
        self._fp = fd

        # TODO verify settings (no compression? not streaming, etc)

        # TODO do I have to write here?
        # now write with junk data
        # if no data is provided the header will be incorrect
        # if I don't call write here, the 'allocated' property appears
        # incorrect. I don't know what this is but it seems to mess
        # up writes for subsequent blocks
        #super().write(fd)

        # fast forward pointer the length of a header + data
        n_bytes = self.header_size + self._data_size
        fd.fast_forward(n_bytes)

        # track the allocation of bytes so subsequent blocks are
        # written directly after this block
        self.allocated = self._size

    def read_chunk(self):
        if not self._written:
            raise KeyError("Block never written")

        # jump to data
        # print(f"read_chunk at data_offset={self.data_offset}")
        self._fp.seek(self.data_offset)

        # read the data but don't memmap or lazy load
        return self._read_data(self._fp, self._size, self._data_size)

    def write_chunk(self, data):
        self._written = True
        self._data = numpy.frombuffer(data, 'uint8', -1)
        self.update_size()
        self._fp.seek(self.offset)
        # print(f"write_chunk at offset={self.offset}")
        super().write(self._fp)


class InternalStore(zarr.storage.KVStore):
    """
    A key-value store for mapping chunks to blocks
    and for storing metadatay (.zarray) in the tree

    Each zarray should have it's own store (so no additional keys are needed)
    """
    def __init__(self, block_manager, zarray=None, block_slice=None):
        # TODO should this be a weak ref?
        self._block_manager = block_manager

        # if a block slice was provided store it for when zarray
        # is set and blocks can be created or read
        self._block_slice = block_slice

        # start with no file pointer
        # TODO not sure if this is needed or if blocks can track this
        #self._fp = None

        # start with no zarray meta data
        # zarray is set, the block size will be known and the
        # asdf_file reference can be used to create the number of blocks
        # of a special ZarrChunkBlock type
        self._zarray = None

        # we will eventually have enough information to make a key
        # value store of blocks that can be looked up based on the
        # chunk key from zarr
        self._block_kv_store = {}

        if zarray is not None:
            self._set_zarray(zarray)

    def _set_zarray(self, zarray_meta):
        self._zarray = zarray_meta
        # TODO check for no compression and other unsupported features
        if self._zarray['compressor'] is not None:
            raise NotImplementedError(
                f"{self.__class__} only supports uncompressed arrays")
        # TODO use dimension seperator other options
        dimension_separator = self._zarray.get('dimension_separator', '.')

        # now that chunk size is known, allocate blocks
        chunk_size = self._chunk_size()

        # make blocks and map them to the internal kv store
        # compute number of chunks (across all axes)
        chunk_counts = [math.ceil(s / c) for (s, c) in
                        zip(self._zarray['shape'], self._zarray['chunks'])]
        # iterate over all chunk keys
        chunk_iter = itertools.product(*[range(c) for c in chunk_counts])

        if self._block_slice is None:
            # save starting block index
            starting_block_index = None
            block_index = None
            for c in chunk_iter:
                key = dimension_separator.join([str(i) for i in c])
                # make a block for each new chunk
                block = ZarrChunkBlock(chunk_size)
                # print(f"Adding block {block} for {key}")
                self._block_manager.add(block)
                new_block_index = self._block_manager.get_source(block)
                if starting_block_index is None:
                    starting_block_index = new_block_index
                else:
                    # verify that blocks are consecutive
                    assert new_block_index == block_index + 1
                block_index = new_block_index
                self._block_kv_store[key] = block
            # make block slice 
            self._block_slice = (starting_block_index, block_index + 1)
        else:  # block slice exists
            block_indices = range(*self._block_slice)
            for (c, i) in zip(chunk_iter, block_indices):
                key = dimension_separator.join([str(i) for i in c])
                block = self._block_manager.get_block(i)
                # convert block to ZarrChunkBlock
                zarr_chunk_block = block_to_zarr_chunk_block(block)
                # replace block with zarr_chunk_block
                i = self._block_manager._internal_blocks.index(block)
                self._block_manager._internal_blocks[i] = zarr_chunk_block
                self._block_kv_store[key] = zarr_chunk_block

    @property
    def block_slice(self):
        return self._block_slice

    def _chunk_size(self):
        dt = numpy.dtype(self._zarray['dtype'])
        n = dt.itemsize
        for c in self._zarray['chunks']:
            n *= c
        return n

    def _chunk_shape(self):
        shape = self._zarray['shape']
        chunks = self._zarray['chunks']
        return [math.ceil(s / c) for (s, c) in zip(shape, chunks)]

    def _get_chunk(self, key):
        # look up the block
        block = self._block_kv_store[key]
        # print(f"_get_chunk key={key}, block={block}")
        # read block
        return block.read_chunk()

    def _set_chunk(self, key, value):
        block = self._block_kv_store[key]
        # print(f"_set_chunk key={key}, block={block}")
        block.write_chunk(value)

    def __getitem__(self, key):
        if key == '.zgroup':
            raise KeyError(".zgroup not found")
        if key == '.zarray':
            if self._zarray is None:
                raise KeyError(".zarray does not exist")
            else:
                return self._zarray
        # chunk
        return self._get_chunk(key)

    def __setitem__(self, key, value):
        if key == '.zarray':
            self._set_zarray(json.loads(value.decode()))
            return
        self._set_chunk(key, value)

    def __delitem__(self, key):
        if key == '.zarray':
            self._zarray = None
            return
        raise NotImplementedError("cannot delete chunks")
