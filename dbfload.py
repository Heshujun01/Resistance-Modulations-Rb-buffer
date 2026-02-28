import os
import numpy as np
import struct
import glob
import logging

# ??????
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("conversion.log"), logging.StreamHandler()]
)

def dbfload(filename, range=None, format='double'):
    """
    Loads data from a DataAcquisition-generated file in a range of indices.
    
    Parameters:
        filename (str): Path to the .dbf file.
        range (list or None): Range of points to load, [start, end] (0-based).
                              If None, loads the full dataset.
        format (str): Output data format, either 'double' or 'int16'.
    
    Returns:
        d (numpy.ndarray): Loaded data.
        h (dict): Header information.
    """
    # Initialize outputs
    d = []
    h = {}

    # Open the file and read header
    try:
        with open(filename, 'rb') as fid:
            # Get total length of file
            file_size = os.path.getsize(filename)

            # Read number of header points and the header
            nh = struct.unpack('I', fid.read(4))[0]  # Read uint32
            hh = fid.read(nh)  # Read header bytes
            h = get_array_from_byte_stream(hh)

            # Calculate start position for data
            fstart = 4 + nh
            bytesize = 2  # int16

            # Determine number of channels and total points
            h['numChan'] = len(h['chNames'])
            h['numTotal'] = (file_size - fstart) / bytesize
            h['numPts'] = h['numTotal'] / h['numChan']

            # Load data or just return header?
            if range == 'info':
                return None, h

            # Determine range to load
            if range is None:
                range = [0, int(h['numPts'])]

            # Seek to the correct position
            fid.seek(fstart + range[0] * bytesize * h['numChan'])

            # Calculate number of points to read
            npts = range[1] - range[0]
            sz = (npts, h['numChan'])

            # Read data as int16
            data_bytes = fid.read(npts * h['numChan'] * bytesize)
            d16 = np.frombuffer(data_bytes, dtype=np.int16).reshape(sz)

            # Check if data is empty
            if d16.size == 0:
                raise ValueError("Data file is empty.")

            # Decide how to output data
            if format == 'int16':
                d = d16
            else:
                # Scale data from int16 to double
                scaling = np.array(h['data_compression_scaling'])
                d = d16.astype(np.float64) / scaling

    except FileNotFoundError:
        raise FileNotFoundError(f"Could not open {filename}")
    except Exception as e:
        raise RuntimeError(f"Error loading file: {e}")

    return d, h


def get_array_from_byte_stream(byte_stream):
    """
    Parses the header byte stream into a dictionary.
    
    Parameters:
        byte_stream (bytes): The header byte stream.
    
    Returns:
        dict: Parsed header information.
    """
    # Placeholder implementation for header parsing
    # You need to implement this based on the actual header structure
    # For example, you can use `struct.unpack` to parse specific fields
    header = {
        'chNames': ['Channel1', 'Channel2'],  # Example channel names
        'data_compression_scaling': [1.0, 1.0]  # Example scaling factors
    }
    return header


def write_abf_file(sweep_data, filename, sample_rate_hz, units='pA'):
    """
    ?? ABF ????????
    
    ??:
        sweep_data (np.ndarray): ?????
        filename (str): ??????
        sample_rate_hz (int): ???(Hz)?
        units (str): ????,??? 'pA'?
    """
    # ????
    BLOCKSIZE = 512
    HEADER_BLOCKS = 4 * 3
    bytes_per_point = 2

    # ??????
    sweep_count = sweep_data.shape[0]
    sweep_point_count = 1  # ???? sweep ?????
    data_point_count = sweep_point_count * sweep_count

    # ?????????????
    data_blocks = int(data_point_count * bytes_per_point / BLOCKSIZE) + 1
    data = bytearray((data_blocks + HEADER_BLOCKS) * BLOCKSIZE)

    # ??????
    struct.pack_into('4s', data, 0, b'ABF ')  # ????
    struct.pack_into('f', data, 4, 1.30)  # ?????
    struct.pack_into('h', data, 8, 3)  # ????(gap-free)
    struct.pack_into('i', data, 10, data_point_count)  # ??????
    struct.pack_into('i', data, 16, sweep_count)  # ?? episodes ??
    struct.pack_into('i', data, 40, HEADER_BLOCKS)  # ?????
    struct.pack_into('h', data, 100, 0)  # ????(??)
    struct.pack_into('h', data, 120, 1)  # ADC ???
    struct.pack_into('f', data, 122, 1e6 / sample_rate_hz)  # ADC ????
    struct.pack_into('i', data, 138, sweep_point_count)  # ?? episode ?????

    # ??????
    f_signal_gain = 1
    f_adc_programmable_gain = 1
    l_adc_resolution = 2**15  # 16 ??????
    max_val = np.max(np.abs(sweep_data))
    f_instrument_scale_factor = 100
    for _ in range(10):
        f_instrument_scale_factor /= 10
        f_adc_range = 10
        value_scale = l_adc_resolution / f_adc_range * f_instrument_scale_factor
        max_deviation_from_zero = 32767 / value_scale
        if max_deviation_from_zero >= max_val:
            break

    # ??????
    unit_string = units.ljust(8)
    struct.pack_into('i', data, 252, l_adc_resolution)
    struct.pack_into('f', data, 244, f_adc_range)
    for i in range(16):
        struct.pack_into('f', data, 922 + i * 4, f_instrument_scale_factor)
        struct.pack_into('f', data, 1050 + i * 4, f_signal_gain)
        struct.pack_into('f', data, 730 + i * 4, f_adc_programmable_gain)
        struct.pack_into('8s', data, 602 + i * 8, unit_string.encode())

    # ??????
    data_byte_offset = BLOCKSIZE * HEADER_BLOCKS
    for sweep_number, sweep_signal in enumerate(sweep_data):
        byte_position = data_byte_offset + sweep_number * sweep_point_count * bytes_per_point
        struct.pack_into('h', data, byte_position, int(sweep_signal * value_scale))

    # ????
    with open(filename, 'wb') as f:
        f.write(data)
    logging.info(f"?????: {filename}")


def dbf2abf_single_file(input_file, freq=50000, voltage_threshold=None):
    """
    ???? .dbf ??? .abf ???
    
    ??:
        input_file (str): ???????
        freq (int): ???,??? 50000 Hz?
        voltage_threshold (float): ????,??? None?
    """
    data, _ = dbfload(input_file)
    data = np.array(data).T
    current = data[0]/32.767
    voltage = data[1]

    if voltage_threshold is not None:
        if voltage_threshold > 0:
            current = np.delete(current, voltage < voltage_threshold)
        else:
            current = -np.delete(current, voltage > voltage_threshold)
        
        logging.info(f"?????? {voltage_threshold} ??????")

    output_file = f"{input_file}.abf"
    write_abf_file(current, output_file, freq)


def dbf2abf_folder(folder, freq=50000, voltage_threshold=None):
    """
    ????????? .dbf ??? .abf ???
    
    ??:
        folder (str): ??????
        freq (int): ???,??? 50000 Hz?
        voltage_threshold (float): ????,??? None?
    """
    engine = start_matlab_engine()
    for file_path in glob.glob(os.path.join(folder, '*.dbf')):
        dbf2abf_single_file(file_path, freq=50000, voltage_threshold=voltage_threshold)


if __name__ == "__main__":
    # ??:??????
    dbf2abf_single_file("example.dbf", freq=50000, voltage_threshold=25)

    # ??:???????????
    dbf2abf_folder("path/to/folder", freq=50000, voltage_threshold=25)