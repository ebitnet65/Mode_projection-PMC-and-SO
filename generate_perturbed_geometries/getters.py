# NOTE: functions that start with "YuZhang" and parts of main() were adopted from
# the script that Dr. Yu Zhang shared with Dr. Bittner's group during his visit in September 2024.
# Therefore, Dr. Yu Zhang should be acknowledged whenever any part of this work is used.

import sys
import numpy as np


# gets the coordinates block and its location (line number) from ORCA input file that was previously read by readlines()
# following the convention, this block is indicated by "*" symbol in the line before and after the block
def get_coord(orca_input_file):
    coord = []
    start_of_coord = -1  # line right before coord. block; must contain "*" symbol
    end_of_coord = -1  # line right after coord. block; must contain "*" symbol
    for i, line in enumerate(orca_input_file):
        if "*" in line:
            start_of_coord = i
            break
    if start_of_coord == -1:  # if there's no "*", exit with error
        print("coordinates not found; line before coord. block must contain *")
        sys.exit()
    for i, line in enumerate(orca_input_file[start_of_coord + 1:]):
        if "*" in line:
            end_of_coord = start_of_coord + 1 + i
            break
        else:
            coord.append(line)
    if end_of_coord == -1:
        print("format error; the line after coord. block must have *")
        sys.exit()
    return coord, start_of_coord, end_of_coord


# reads g-tensor from ORCA output file
def YuZhang_get_gtensor(fname):
    logdata = fname
    g = np.empty([3, 3])
    for k, line in enumerate(logdata):
        if "The g-matrix:" in line:
            for i in range(3):
                gtemp = logdata[k + i + 1].split()
                g[i] = [float(g) for g in gtemp]
            break
    return g


# reads dipole moment, g-tesor and HFC matrix from ORCA output
def YuZhang_get_dipole_g_hfc(orca_output):
    for k, line in enumerate(orca_output):
        # get dipole moment
        if "Total Dipole Moment" in line:
            dipole0 = [float(line.split()[k]) for k in range(4, 7)]

        # get gtensor
        if "The g-matrix:" in line:
            g = np.empty([3, 3])
            for i in range(3):
                gtemp = orca_output[k + i + 1].split()
                g[i] = [float(g) for g in gtemp]

        # get hypferfine parameter
        hfc = np.zeros([3, 3])
        if "Raw HFC matrix " in line:
            # print(line)
            hfc = np.empty([3, 3])
            for i in range(3):
                gtemp = orca_output[k + i + 2].split()
                hfc[i] = [float(g) for g in gtemp]
    return dipole0, g, hfc


# calculates the derivative for a 3x3 matrix using the central point difference scheme
def YuZhang_gradient_sub(A2, A1, dx):
    gradient = np.empty([3, 3])
    for k in range(3):
        for j in range(3):
            gradient[k][j] = (A2[k][j] - A1[k][j]) / 2.0 / dx
    return gradient


# calculates the derivative for a vector of length 3 using the central point difference scheme
def YuZhang_gradient_sub2(A2, A1, dx):
    gradient = np.empty([3])
    for k in range(3):
        gradient[k] = (A2[k] - A1[k]) / 2.0 / dx
    return gradient


# reads the output of ORCA !Freq run:
# reads the number of frequencies, frequencies themselves, and (weighed) normal modes (Lis matrix)
# NOTE: this function does not change units
def get_vibrational_data(orca_freq_output):
    freq = []
    normal_modes_unsorted = []
    start_of_freq = -1  # line right before coord. block
    start_of_modes = -1  # line right before normal modes' block
    for i, line in enumerate(orca_freq_output):
        if "VIBRATIONAL FREQUENCIES" in line:
            start_of_freq = i + 4
        if "NORMAL MODES" in line:
            start_of_modes = i + 6

    # prompts an error message if no data was found
    if start_of_freq == -1:
        print("frequencies not found")
        sys.exit()
    if start_of_modes == -1:
        print("normal modes not found")
        sys.exit()

    # gets the frequencies, change the type into float
    for i, line in enumerate(orca_freq_output[start_of_freq + 1:]):
        if not line.strip():  # stops at the end of frequency block which is signified by a blank line
            break
        else:
            # Append only the 2nd element of each line, which is the frequency value itself,
            # and omit the 1st element, which is the corresponding frequency's index number.
            freq.append(float(line.split()[1]))

    # number of modes is equal to the total number of frequencies
    number_of_modes = len(freq)

    # gets the normal modes' portion as a 1D list
    for i, line in enumerate(orca_freq_output[start_of_modes + 1:]):
        if not line.strip():  # -//-
            break
        else:
            normal_modes_unsorted.append(line)

    # a multistep process that produces normal modes matrix (Lis) from normal modes' portion of orca output
    # it breaks the normal_modes_unsorted into "chunks" and rearranges them to form the actual matrix
    chunk_size = 1 + number_of_modes  # +1 since there are extra lines that contain index numbers
    number_of_chunks = len(normal_modes_unsorted) // chunk_size

    normal_modes = np.empty((number_of_modes, 0))

    for i in range(number_of_chunks):
        chunk_float = []
        chunk = normal_modes_unsorted[:chunk_size]

        # removes a portion of the array that is no longer needed
        normal_modes_unsorted = normal_modes_unsorted[chunk_size:]

        # removes the line with index numbers
        del chunk[0]

        for item in chunk:
            chunk_float.append([float(x) for x in item.split()][1:])

        # converts to numpy array to make it easier to reshape it
        chunk_float_np = np.array(chunk_float)

        chunk_float_np_reshaped = np.reshape(chunk_float_np, (number_of_modes, -1))
        normal_modes = np.append(normal_modes, chunk_float_np_reshaped, axis=1)

    # makes sure that the final Lis matrix is actually square
    if normal_modes.shape[0] != normal_modes.shape[1]:
        print("Normal modes matrix (Lis) is not square!")
        sys.exit()

    return number_of_modes, freq, normal_modes
