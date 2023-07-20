import math
import os

import numpy as np


def serialize(submission: list[np.ndarray], path_or_filehandle):
    # File format, where "pi" is the i-th prediction array and parenthesized terms indicate the number of bits/bytes
    # max_len_bytes (uint8)
    # len_p1 (max_len_bytes) p1 (len_p1 * uint8) ... len_pn (max_len_bytes) pn (len_pn * uint8)
    
    # Get the maximum prediction length and the bytes required to store this length. We will store the actual lengths
    # with an offset by 1, e.g., for some prediction array "pi", len(pi) = 250 --> len_pi = 249, to avoid needing an
    # additional byte to store lengths that are certain powers of 2 ("byte powers"), e.g., len(pi) = 2^8 (1 byte) = 256
    # --> len_pi = 255 (the number 256 (literally) would require 2 bytes (9 bits)).
    max_len = max(len(p) for p in submission)
    max_len_bytes = math.ceil(math.log2(max_len) / 8)
    
    def write(f):
        # Write the number of bytes that are needed to store the length of each prediction array as a fixed unsigned
        # integer with 8 bits. This means that the maximum supported number of bytes per length entry is 2^8 - 1 = 255,
        # which should be enough in basically all real-world use cases, since the actual length entries could then be
        # numbers up to 2^(255 * 8).
        f.write(max_len_bytes.to_bytes(length=1, byteorder="big", signed=False))
        for prediction in submission:
            if prediction.dtype != np.uint8:
                raise TypeError("all arrays must be of data type np.uint8")
            if prediction.ndim != 1:
                raise ValueError("all arrays must be 1D")
            # Write the length of the following prediction array (with offset by 1)
            length = len(prediction) - 1
            f.write(length.to_bytes(length=max_len_bytes, byteorder="big", signed=False))
            # Write the prediction array
            f.write(prediction.tobytes())
    
    # If it is a path to file, wrap the write function into a with-open context manager
    if isinstance(path_or_filehandle, (str, bytes, os.PathLike)):
        with open(path_or_filehandle, "wb") as fh:
            return write(fh)
    # Otherwise, assume it is already a filehandle, in which case the user is responsible for proper file closing
    return write(path_or_filehandle)


def deserialize(path_or_filehandle) -> list[np.ndarray]:
    def read(f):
        submission = []
        # Read the first byte, which represents the number of bytes that each of the following prediction array lengths
        # requires (see serialization process above)
        max_len_bytes = int.from_bytes(f.read(1), byteorder="big", signed=False)
        
        while True:
            # Read the length of the following prediction array
            length_bytes = f.read(max_len_bytes)
            # If there is nothing left (EOF), we processed the entire file. If the serialization was correct and without
            # any errors, the not-EOF means that there is guaranteed to be a length entry and then the corresponding
            # prediction array afterward, i.e., it's either length+array or EOF, and nothing in between. If, for some
            # reason, this is not the case (maybe corrupt data), then either the int.from_bytes or np.frombuffer will
            # fail and, in turn, the entire deserialization.
            if not length_bytes:
                return submission
            # We serialized the length with an offset by 1, so add it again to get the true length
            length = int.from_bytes(length_bytes, byteorder="big", signed=False) + 1
            # Read the prediction array, i.e., the file content from the current byte position c until c+length. This
            # only works because we know that every element is of data type np.uint8 = 1 byte, so the entire array must
            # be length bytes.
            prediction = np.frombuffer(f.read(length), dtype=np.uint8)
            submission.append(prediction)
    
    # If it is a path to file, wrap the read function into a with-open context manager
    if isinstance(path_or_filehandle, (str, bytes, os.PathLike)) and os.path.isfile(path_or_filehandle):
        with open(path_or_filehandle, "rb") as fh:
            return read(fh)
    # Otherwise, assume it is already a filehandle, in which case the user is responsible for proper file closing
    return read(path_or_filehandle)
