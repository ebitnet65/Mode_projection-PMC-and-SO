# NOTE: functions that start with "YuZhang" and parts of main() were adopted from
# the script that Dr. Yu Zhang shared with Dr. Bittner's group during his visit in September 2024.
# Therefore, Dr. Yu Zhang should be acknowledged whenever any part of this work is used.

from generate_perturbed_geometries.getters import *
from generate_perturbed_geometries.atom import *
import os

# declaration of physical constants
h = 6.62607015e-34
c = 2.99792458e10
e = 1.60217663e-19
me = 9.1093837015e-31
mp = 1.4106067974e-26
pi = 3.1415926e0

# conversion coefficients
cm2ev = c * h / e
ev2au = e / 4.3597447222071e-18
cm2au = cm2ev * ev2au
mp_au = mp / me

# perturbation distance used in the simulations
perturb_distance = 0.01

# path to results from cluster
prepath = "../from_cluster"
which_setup = "/results_tzvp"
cluster_path = prepath + which_setup

# choose main prefix
main_prefix = "so"

# path to the original unperturbed optimized geometry input file
path_to_input = cluster_path + '/epr_orca/' + main_prefix + '/system.inp'
with open(path_to_input, "r") as f:
    orca_input = f.read().splitlines()

# path to the output of !Freq ORCA calculation
path_to_vib = cluster_path + '/vibrational_analysis/' + main_prefix + '/std.out'
with open(path_to_vib, "r") as f:
    vib_output = f.read().splitlines()

# path to an EPR calculation on the unperturbed geometry
path_to_baseline = cluster_path + '/epr_orca/' + main_prefix + '/std.out'
with open(path_to_baseline, "r") as f:
    baseline_output = f.read().splitlines()

# directory with all the separate atom folders
# NOTE: this script assumes the following directory tree which must be present:
# /dirs/1/ ; /dirs/2/ ... /dirs/"n=number_of_atoms"/
#   /dirs/1/x/ ; /dirs/1/y/ ; /dirs/1/z/ ; /dirs/2/x/ ...
#       /dirs/1/x/plus/ ; /dirs/1/x/minus/ ; /dirs/1/y/plus/ ...
path_to_perturb = cluster_path + "/output_perturbations/" + main_prefix + "/dirs"

# output top directory
path_to_output = "./input_for_qutip/" + main_prefix


# calculate the spin-phonon coupling
def YuZhang_sph_coupling(nmode, vib_freq, normal_modes, dg):
    dgq = []
    for m in range(nmode):
        freq = vib_freq[m]
        dq = np.zeros((3, 3))

        for k in range(nmode):
            gradient = dg[k]
            Lis = normal_modes[m][k]

            # mi = mass[iat]
            #     ( hbar)           d H
            # sqrt(-----) L^a_{is} ----
            #     (w m_i)           dXis

            tmp = 0.0
            if freq > 1.e0:
                tmp = math.sqrt(1.0 / (freq * cm2au * mp_au)) * Lis

            for i in range(3):
                for j in range(3):
                    dq[i][j] = dq[i][j] + tmp * gradient[i][j]

        dgq.append(dq)

    return dgq


def main():
    # declaration of arrays following YuZhang notation:
    # dd - gradients of dipole
    # dg - gradients of g-tensor
    # da - gradients of HFC matrices
    dd = []
    dg = []
    da = []

    # get the information from the original input file
    initial_coord, st, en = get_coord(orca_input)

    # get the number of atoms in the molecule
    number_of_atoms = len(initial_coord)

    # calculate the gradients of dipole moment, g-tensor and HFC matrix
    for i in range(number_of_atoms):
        for j in ["x", "y", "z"]:
            path_tmp = path_to_perturb + "/" + str(i + 1) + "/" + j + "/"
            path_minus = path_tmp + "minus/std.out"
            path_plus = path_tmp + "plus/std.out"
            with open(path_minus, "r") as f:
                std_minus = f.read().splitlines()
            with open(path_plus, "r") as f:
                std_plus = f.read().splitlines()

            dipole_minus, g_minus, hfc_minus = YuZhang_get_dipole_g_hfc(std_minus)
            dipole_plus, g_plus, hfc_plus = YuZhang_get_dipole_g_hfc(std_plus)

            dipole_grad = YuZhang_gradient_sub2(dipole_plus, dipole_minus, perturb_distance)
            g_grad = YuZhang_gradient_sub(g_plus, g_minus, perturb_distance)
            hfc_grad = YuZhang_gradient_sub(hfc_plus, hfc_minus, perturb_distance)

            dd.append(dipole_grad)
            dg.append(g_grad)
            da.append(hfc_grad)

    # get vibrational data
    nmode, vib_freq, normal_modes = get_vibrational_data(vib_output)

    # Get the main component of spin-phonon couplings.
    # This component is obtained by converting derivatives of g-tensors for each atom in Cartesian space
    # into derivatives with respect to normal modes.
    # "dgq" is the converted g-tensor, "daq" - HFC matrix.
    dgq = YuZhang_sph_coupling(nmode, vib_freq, normal_modes, dg)
    daq = YuZhang_sph_coupling(nmode, vib_freq, normal_modes, da)

    # read the g-tensor of unperturbed system
    g_tensor = YuZhang_get_gtensor(baseline_output)

    # The following output format is compatible with Dr. Nosheen Younas's Jupyter notebooks
    # available at https://github.com/NosheenYounas/Spin-Phonon-Dynamics-1-2/blob/main

    # write positive normal modes and corresponding derivatives of g-tensor
    foutput = open(path_to_output + "/" + 'spin_phonon.dat', 'w')
    foutput.write('unit: cm^{-1} \n')
    for k, freq in enumerate(vib_freq):
        dg = dgq[k]
        if freq < 1.e0: continue
        foutput.write('mode: %4d freq=%10.4f \n' % (k, freq))
        for i in range(3):
            foutput.write('%15.7e %15.7e %15.7e \n' % (dg[i][0], dg[i][1], dg[i][2]))
    foutput.close()

    # write g-tensor
    foutput = open(path_to_output + "/" + 'gtensor.dat', 'w')
    foutput.write('g tensor\n')
    for i in range(3):
        foutput.write('%15.7e %15.7e %15.7e \n' % (g_tensor[i][0], g_tensor[i][1], g_tensor[i][2]))
    foutput.write('\nhpc matrix\n')
    hfc = np.zeros([3, 3])  # dummy matrix, kept for compatibility
    for i in range(3):
        foutput.write('%15.7e %15.7e %15.7e \n' % (hfc[i][0], hfc[i][1], hfc[i][2]))
    foutput.close()

    # generate and write mass-unscaled output; requires system's geometry data in .xyz format
    if os.path.exists(path_to_output + "/" + 'optimized.xyz'):
        my_xyz = read_xyz(path_to_output + "/" + 'optimized.xyz')
        my_masses_sqrt = [np.sqrt(atom.atomtype.weight) for atom in my_xyz]
        my_masses_sqrt_triple = np.repeat(my_masses_sqrt, 3)
        mass_sqrt_matrix = np.diag(my_masses_sqrt_triple)
        normal_modes_unscaled = np.matmul(mass_sqrt_matrix, normal_modes)

        # write normal modes and corresponding mass-unscaled eigenvectors
        foutput = open(path_to_output + "/" + 'spin_phonon_bare_unscaled.dat', 'w')
        foutput.write('unit: cm^{-1} \n')
        for k, freq in enumerate(vib_freq):
            dg = dgq[k]
            if freq < 1.e0: continue
            foutput.write('mode %4d freq=%10.4f \n' % (k, freq))
            foutput.write(" ".join([str(x) for x in normal_modes_unscaled[:, k]]))
            foutput.write("\n")
            for i in range(3):
                foutput.write('%15.7e %15.7e %15.7e \n' % (dg[i][0], dg[i][1], dg[i][2]))
        foutput.close()

    # write normal modes and corresponding mass-scaled eigenvectors
    foutput = open(path_to_output + "/" + 'spin_phonon_bare.dat', 'w')
    foutput.write('unit: cm^{-1} \n')
    for k, freq in enumerate(vib_freq):
        dg = dgq[k]
        if freq < 1.e0: continue
        foutput.write('mode %4d freq=%10.4f \n' % (k, freq))
        foutput.write(" ".join([str(x) for x in normal_modes[:, k]]))
        foutput.write("\n")
        for i in range(3):
            foutput.write('%15.7e %15.7e %15.7e \n' % (dg[i][0], dg[i][1], dg[i][2]))
    foutput.close()


if __name__ == "__main__":
    main()
