import itertools
import json
import math

import asdf
import numpy
import zarr


def block_to_zarr_chunk_block(block):
    zarr_chunk_block = ZarrChunkBlock(block._data_size)

    # since we're working with a pre-allocated block that has data
    # we can assume it's '_written' for ZarrChunkBlock purposes
    zarr_chunk_block._written = True

    # copy over the file pointer
    zarr_chunk_block._fp = block._fd

    # and the position relative to the start of the file
    zarr_chunk_block.offset = block.offset
    return zarr_chunk_block


class ZarrChunkBlock(asdf.block.Block):
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
        self._fp.seek(self.data_offset)

        # read the data but don't memmap or lazy load
        return self._read_data(self._fp, self._size, self._data_size)

    def write_chunk(self, data):
        # mark this chunk as 'written' so reads will succeed
        self._written = True

        # convert data to bytes format
        self._data = numpy.frombuffer(data, 'uint8', -1)

        # jump to the position in the file where the block is stored
        self._fp.seek(self.offset)

        # write the block to the file
        super().write(self._fp)


class InternalStore(zarr.storage.KVStore):
    """
    A key-value store for mapping chunks to blocks
    and for storing metadata (.zarray) in the tree

    Each zarray should have it's own store (so no additional keys are needed)
    """
    def __init__(self, block_manager, zarray=None, block_slice=None):
        # TODO should this be a weak ref?
        self._block_manager = block_manager

        # if a block slice was provided store it for when zarray
        # is set and blocks can be created or read
        self._block_slice = block_slice

        # start with no zarray meta data
        # zarray is set, the block size will be known and the
        # asdf_file reference can be used to create the number of blocks
        # of a special ZarrChunkBlock type
        self._zarray = None

        # we will eventually have enough information to make a key
        # value store of blocks that can be looked up based on the
        # chunk key from zarr
        self._block_kv_store = {}

        # if zarray meta data was provided, initialize the store
        if zarray is not None:
            self._set_zarray(zarray)

    def _set_zarray(self, zarray_meta):
        # copy the meta data for use in computing chunks etc
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

                # add the new block to the block manager
                self._block_manager.add(block)

                # get the index for the new block
                new_block_index = self._block_manager.get_source(block)
                if starting_block_index is None:
                    starting_block_index = new_block_index
                else:
                    # verify that blocks are consecutive
                    assert new_block_index == block_index + 1
                block_index = new_block_index
                self._block_kv_store[key] = block

            # the _block_slice are the block indices within which chunks are
            # stored, this will be written to the tree in the source key
            self._block_slice = (starting_block_index, block_index + 1)
        else:  # block slice exists, so we're reading from a file with blocks
            block_indices = range(*self._block_slice)
            for (c, bi) in zip(chunk_iter, block_indices):
                key = dimension_separator.join([str(i) for i in c])

                # look up the Block instance at an index
                block = self._block_manager.get_block(bi)

                # convert block to ZarrChunkBlock
                zarr_chunk_block = block_to_zarr_chunk_block(block)

                # replace block with zarr_chunk_block
                i = self._block_manager._internal_blocks.index(block)
                self._block_manager._internal_blocks[bi] = zarr_chunk_block

                # store the mapping between blocks and zarr chunk keys
                self._block_kv_store[key] = zarr_chunk_block

    @property
    def block_slice(self):
        return self._block_slice

    def _chunk_size(self):
        """Compute the size, in bytes, of a chunk"""
        dt = numpy.dtype(self._zarray['dtype'])
        n = dt.itemsize
        for c in self._zarray['chunks']:
            n *= c
        return n

    def _chunk_shape(self):
        """Compute the shape (number per dimension) of the chunks"""
        shape = self._zarray['shape']
        chunks = self._zarray['chunks']
        return [math.ceil(s / c) for (s, c) in zip(shape, chunks)]

    def _get_chunk(self, key):
        """Given a zarr chunk key, look up and read the block"""
        # look up the block
        block = self._block_kv_store[key]
        # read block
        return block.read_chunk()

    def _set_chunk(self, key, value):
        """Given a zarr chunk key, look up and write to the block"""
        block = self._block_kv_store[key]
        block.write_chunk(value)

    def __getitem__(self, key):
        if key == '.zgroup':
            raise KeyError(".zgroup not found")
        if key == '.zarray':
            if self._zarray is None:
                raise KeyError(".zarray does not exist")
            else:
                return self._zarray
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

    def write_remaining_blocks(self):
        for block in self._block_kv_store.values():
            if not block._written:
                # TODO what is a sensible default? or should we throw an
                # exception if no fill value is defined
                if 'fill_value' not in self._zarray:
                    raise Exception(f"Unwritten block {block=} and undefined fill_value")
                v = self._zarray['fill_value']
                data = numpy.empty(self._zarray['chunks'], self._zarray['dtype'])
                data[:] = v
                block.write_chunk(data)
