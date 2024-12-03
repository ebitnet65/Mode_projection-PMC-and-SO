from readfiles import *
import numpy.linalg as LA
from unit import *
from qutip import *
import numpy as np
from scipy.optimize import curve_fit

# line broadening for spectral density, in meV
SIG = 2.0

# temperature Scan, multiple runs, in K
Temp_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320]

# time intervals for numerical integration of Bloch-Redfield equation, in ps
tlist = np.linspace(1, 1000e6, 10000)
tlist_t2 = np.linspace(1, 100e6, 200000)

# list of amplitudes of magnetic field, in Tesla
bfield_list = [0.1, 0.2, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# corresponding list of output folders
folder_list = [str(x + 1) for x in range(10)]

# loading the gtensor for the basic Hamiltonian.
gtensor_fname = 'data_files/gtensor.dat'
sp_fname = 'data_files/spin_phonon.dat'

# Now defining Units and conversion factors.
h = 6.62607015e-34  # planck's constant in joules second
c = 2.99792458e10  # speed of light in cm per second
e = 1.60217663e-19  # charge of electron in coulombs
hbar = 0.658211951  # in eV-fs or meV-ps or ueV-ns
cm2ev = c*h/e  # conversion factor from 1/cm to eV
cm2mev = cm2ev * 1.e3  # conversion from 1/cm to meV
ev2au = e/4.3597447222071e-18  # converison from eV to atomic mass units
cm2au = cm2ev * ev2au  # conversion from 1/cm to atomic mass units

kB = 8.617333262e-2  # meV/K, Boltzmann constant


free_e_gyro = 28024.9514242E6  # Hz/T gyromagnetic ratio of an electron.
hz2meV = 4.1357E-12  # 1Hz = 4.1357E-15 eV = 4.1357E-12 meV
free_e_gfactor = 2.00231930437378  # free electron g factor 2.00231930437378
alpha = (free_e_gyro*hz2meV)/free_e_gfactor  # gyromagnetic ratio/gfactor in meV per tesla


# returns a Gaussian/Normal distribution centered at w0 with std sigma
def shape_normal(w, w0, sigma):
    return np.exp(-0.5*((w-w0)/sigma)**2)/(sigma*np.sqrt(2*np.pi))


# returns a lorentzian profile centered at 'w0' with thickness 'sigma'
# NOTE: sigma is related to the lifetime of the mode by tau = hbar/sigma
def lorentzian(w, w0, sigma):
    return sigma/(np.pi*(sigma**2 + (w-w0)**2))


# n = n_thermal = pop of harmonic oscillator mode with frequency ‘w’,
# at the temperature described by ‘w_th’ where w_th = kBT/hbar
# NOTE: frequencies and couplings must be passed in energy units i.e. meV; units of S(w) are [(meV)^2 s]
def spectral_density_quantum(w, freqs, coup, T=5, sigma=SIG, shape='L'):
    kB = 8.6173e-2  # meV/K, Boltzmann constant
    w_th = kB*T/hbar  # temperature defined as frequency
    S = 0
    for j in range(len(freqs)):
        wj = freqs[j]/hbar
        cj = coup[j]/np.sqrt(hbar/(2*wj)) # energy / length
        nj = n_thermal(wj,w_th)
        if shape == 'G':
            S += np.pi*hbar*(cj**2/wj)*( (nj+1)*shape_normal(w, wj, sigma) + nj*shape_normal(w, -wj, sigma) )
        else:
            S += np.pi*hbar*(cj**2/wj)*( (nj+1)*lorentzian(w, wj, sigma) + nj*lorentzian(w, -wj, sigma) )
    return S


# units of S(w) are energy (meV) if frequencies and couplings are in frequency units
def spectral_density_classical(w, freqs, coup, T=5, sigma=SIG, shape='L'):
    kB = 8.6173e-2 # meV/K, Boltzmann constant
    S = 0
    for j in range(len(freqs)):
        wj = freqs[j]/hbar
        cj = coup[j]/np.sqrt(hbar/(2*wj)) # energy/length
        if shape=='G':
            S += np.pi*kB*T*(cj**2/wj**2)*(shape_normal(w, wj, sigma) + shape_normal(w, -wj, sigma))
        else:
            S += np.pi*kB*T*(cj**2/wj**2)*(lorentzian(w, wj, sigma) + lorentzian(w, -wj, sigma))
    return S


# fitting functions for T1 and T2
def t1_func(x, t1, xinf):
    return (1 - xinf) * np.exp(-x / t1) + xinf


def t2_func(x,t2):
    return np.exp(-x/t2)


def t1_func_classical(x, t1):
    return (1 / 2) * np.exp(-x / t1) + (1 / 2)


# get the Hamiltonian and replaces it with its diagonal form assuming that off-diagonal elements are negligibly small
# this allows for a much faster calculation/convergence
def get_hamiltonian(gtensor, sigma_vec, Bfield):
    Bvector = np.zeros(3)
    Bvector[0] = 0
    Bvector[1] = 0
    Bvector[2] = Bfield
    # Defining spin Hamiltonian
    H = None

    # Zeeman interaction
    for i in range(3):
        for j in range(3):
            if H is None:
                # the "1/2" factor appears here since the pauli matrices are being used instead of spin 1/2 matrices
                H = alpha * gtensor[i][j] * (1/2) * sigma_vec[i] * Bvector[j]
            else:
                H += alpha * gtensor[i][j] * (1/2) * sigma_vec[i] * Bvector[j]

    energies, estates = (-H).eigenstates()
    H = H.transform(estates)
    return H


# get the Hamiltonian, keep off-diagonal terms
def get_hamiltonian2(gtensor, sigma_vec, Bfield):
    Bvector = np.zeros(3)
    Bvector[0] = 0
    Bvector[1] = 0
    Bvector[2] = Bfield
    # Defining spin Hamiltonian
    H = None

    # Zeeman interaction
    for i in range(3):
        for j in range(3):
            if H is None:
                # the "1/2" factor appears here since the pauli matrices are being used instead of spin 1/2 matrices
                H = alpha * gtensor[i][j] *(1/2)* sigma_vec[i] * Bvector[j]
            else:
                H += alpha * gtensor[i][j] *(1/2)* sigma_vec[i] * Bvector[j]
    return H


# a filter that cuts off potentially diverging tail (happens due to numerical instability)
def cut_up_to_min_global(x, y, direction='left_to_right'):
    global_min_index = np.argmin(y)
    if direction == 'right_to_left':
        return x[global_min_index:], y[global_min_index:]
    else:
        return x[:global_min_index], y[:global_min_index]


# calculates relaxation time T1 at given temperatures Temp_list
def get_t1(Temp_list, freq, gcoupS, tlist, H):

    t1_list = np.array([])
    t1d_list = np.array([])

    rho0 = basis(2, 0) * basis(2, 0).dag()
    e_ops = []
    rho_list = []
    options = Options()
    options.nsteps = 5000

    for k in range(len(Temp_list)):
        temp = Temp_list[k]
        print("calculating T1 for ", temp, " K")
        specDensityX = lambda w: spectral_density_quantum(w, freqs=freq, coup=gcoupS[:, 0], T=temp, sigma=SIG,
                                                          shape='L') / hbar ** 2
        specDensityY = lambda w: spectral_density_quantum(w, freqs=freq, coup=gcoupS[:, 1], T=temp, sigma=SIG,
                                                          shape='L') / hbar ** 2
        specDensityZ = lambda w: spectral_density_quantum(w, freqs=freq, coup=gcoupS[:, 2], T=temp, sigma=SIG,
                                                          shape='L') / hbar ** 2
        aops = [[sigmax(), specDensityX], [sigmay(), specDensityY], [sigmaz(), specDensityZ]]

        results_k = brmesolve(H / (hbar), rho0, tlist, a_ops=aops, e_ops=e_ops, progress_bar=True)
        rho_list.append(results_k.states)  # saving states for later
        rho11_k = np.array([results_k.states[j].full()[0, 0] for j in range(len(tlist))])  # first element of rho vs time
        f, df = curve_fit(t1_func, tlist / 1e6, np.abs(rho11_k))
        t1, t1d = f[0], df[0, 0]
        t1_list = np.append(t1_list, t1)
        t1d_list = np.append(t1d_list, t1d)

    return t1_list, t1d_list


# calculates dephasing time T2 at given temperatures Temp_list
def get_t2(Temp_list, freq, gcoupS, tlist_t2, H):
    t2_list = np.array([])
    t2d_list = np.array([])

    rho0 = basis(2, 0) * basis(2, 1).dag()
    e_ops = []
    rho_list = []

    for k in range(len(Temp_list)):
        temp = Temp_list[k]
        print("calculating T2 for ", temp, " K")

        specDensityX = lambda w: spectral_density_quantum(w, freqs=freq / hbar, coup=gcoupS[:, 0] / hbar, T=temp, sigma=SIG)
        specDensityY = lambda w: spectral_density_quantum(w, freqs=freq / hbar, coup=gcoupS[:, 1] / hbar, T=temp, sigma=SIG)
        specDensityZ = lambda w: spectral_density_quantum(w, freqs=freq / hbar, coup=gcoupS[:, 2] / hbar, T=temp, sigma=SIG)
        aops = [[sigmax(), specDensityX], [sigmay(), specDensityY], [sigmaz(), specDensityZ]]

        results_k = brmesolve(H / (hbar), rho0, tlist_t2, a_ops=aops, e_ops=e_ops, progress_bar=True)
        rho_list.append(results_k.states)  # saving states for later
        rho12_k = np.array(
            [results_k.states[j].full()[0, 1] for j in range(len(tlist_t2))])  # first element of rho vs time
        tlist_cut, rho_cut = cut_up_to_min_global(tlist_t2, np.abs(rho12_k))
        t2, t2d = curve_fit(t2_func, tlist_cut / 1e6, np.abs(rho_cut))
        t2_list = np.append(t2_list, t2)
        t2d_list = np.append(t2d_list, t2d)
    return t2_list, t2d_list


# calculate spin-phonon couplings at given magnetic filed Bfield
def get_gcoupS(Bfield, dgx, freq):
    Bvector = np.zeros(3)
    Bvector[0] = 0
    Bvector[1] = 0
    Bvector[2] = Bfield
    Np = len(freq)
    gcoupS = np.zeros((Np, 3), dtype=float)  # defining a placeholder for couplings
    for k in range(Np):
        for j in range(3):
            gcoupS[k, :] += alpha * dgx[k, :, j] * Bvector[j] * (1 / 2)
            # only thing missing from here is Pauli matrix. Everything else is contained.
            # Therefore, couplings will have units of energy as meV

    # NOTE: this coupling also has the harmonic bath displacement operator contained in it.
    # Therefore, it has units of meV.
    return gcoupS


# calculate spin-phonon couplings at given magnetic filed Bfield, suitable for SVD
def get_gcoupS_svd(Bfield, dgx, freq):
    Bvector = np.zeros(3)
    Bvector[0] = 0
    Bvector[1] = 0
    Bvector[2] = Bfield
    Np = len(freq)
    gcoupS = np.zeros((Np, 3), dtype=float)  # defining a place holder for couplings
    for k in range(Np):
        for j in range(3):
            gcoupS[k, :] += alpha * dgx[k, :, j] * Bvector[j] * (1 / 2) * np.sqrt(hbar / (2 * freq[k] / hbar))
    return gcoupS


# write coordinate file with normal modes acting as "forces", following .xsf XCrysDen format
def write_normalmode_xsf(coords, normal_modes, printlist, Usys=None):
    from periodictable import elements
    nats = len(coords)

    if Usys is not None:
        nmode, nsys = Usys.shape

    for k in printlist:
        if Usys is None:
            output = open('xsf/normalmode_%d.xsf' % k, 'w')
            vec = normal_modes[k]
        else:
            output = open('xsf/proj_normalmode_%d.xsf' % k, 'w')
            vec = np.zeros(3 * nats)
            for i in range(nmode):
                vec += normal_modes[i] * Usys[i, k].real
        output.write("ATOMS\n")

        j = 0
        for i in range(nats):
            symbol = coords[i][0]
            artnumber = elements.symbol(symbol).number
            vx = vec[j]
            vy = vec[j + 1]
            vz = vec[j + 2]
            j += 3
            output.write(
                "{}  {}  {}  {} {}  {}  {}\n".format(artnumber, coords[i][1], coords[i][2], coords[i][3], vx, vy, vz))
        output.close()


# get the coordinates and normal modes
def get_modes2(fname, fgeo, nmode=None):
    # load coordinates
    logdata = open(fgeo).readlines()
    nats = int(logdata[0])
    coords = []
    for i in range(2, 2 + nats):
        xyz = logdata[i].split()
        coords.append(xyz)

    logdata = open(fname).readlines()
    logdata = logdata[1:]

    normal_modes = np.zeros((nmode, 3 * nats))
    for i in range(nmode):
        j = i * 5 + 1
        line = logdata[j]
        mode_vec = line.split()
        mode_vec = np.array(mode_vec)
        mode_vec = mode_vec.astype(float)
        normal_modes[i, :] = mode_vec

    return coords, normal_modes


def main():

    # read g-tensor
    gtensor, hfc, zfs = read_g(gtensor_fname)

    # generate pauli matrices
    eS = jmat(1 / 2)
    sigma_vec = [sigmax(), sigmay(), sigmaz()]

    # loading the couplings and frequencies
    freq, dgx = get_phonon(sp_fname)
    freq = freq * cm2mev

    Np = len(freq)

    for ii in range(len(bfield_list)):
        # calculate T1 and T2 (t1_list and t2_list, correspondingly), with error bars (t1d_list nad t2d_list)

        Bfield = bfield_list[ii]  # Tesla
        folder_name = folder_list[ii]

        gcoupS = get_gcoupS(Bfield, dgx, freq)

        H = get_hamiltonian(gtensor, sigma_vec, Bfield)

        t1_list, t1d_list = get_t1(Temp_list, freq, gcoupS, tlist, H)
        t2_list, t2d_list = get_t2(Temp_list, freq, gcoupS, tlist_t2, H)

        # lines below can be used to write down the outputs
        # main_prefix = 'pmc'
        # np.save('saved_variables/' + main_prefix + '/' + folder_name + '/t1_list', t1_list)
        # np.save('saved_variables/' + main_prefix + '/' + folder_name + '/t1d_list', t1d_list)
        #
        # np.save('saved_variables/' + main_prefix + '/' + folder_name + '/t2_list', t2_list)
        # np.save('saved_variables/' + main_prefix + '/' + folder_name + '/t2d_list', t2d_list)

        # projected modes technique

        # performing SVD
        gcoupTest = get_gcoupS_svd(Bfield, dgx, freq)

        U, L, Vdag = LA.svd(gcoupTest, full_matrices=False)  # note: the output of this funciton is U*S*Vh
        print('\n singular values are ', L)

        # Collecting columns of U that correspond to non-zero singular values
        threshold = 1.e-12
        Lnonzero = np.where(L > threshold)[0]  # indices where the singular value is non-zero
        Np_new = len(Lnonzero)  # total nonzero singular values

        print('\n indices of non-zero singular values=', Lnonzero)
        print('\n number of non-zeor singular values=', Np_new)

        Usys = U[:, Lnonzero]
        print('shape of Usys', np.shape(Usys))

        # defining projection operators with the SVD Usys
        A = Qobj(Usys)
        # A = Qobj(gcoupS) # regular mode projection
        P = A * (A.dag() * A).inv() * A.dag()
        I = qeye(Np)
        Q = I - P

        print('shape of projection operator is ', np.shape(P))

        Omega = Qobj(np.diag(freq ** 2))  # Hessian
        Omega_s = P * Omega * P
        Omega_b = Q * Omega * Q
        val_s, vecs_s = Omega_s.eigenstates()
        val_b, vecs_b = Omega_b.eigenstates()

        Pnonzero = np.where(val_s > 1.e-9)[0]
        Qnonzero = np.where(val_b > 1.e-9)[0]

        # collecting nonzero-eigenvalues^0.5 of omega_s and omega_b
        omega_s = np.real(np.sqrt(val_s[Pnonzero]))
        omega_b = np.real(np.sqrt(val_b[Qnonzero]))
        Ks = vecs_s[Pnonzero]
        Kb = vecs_b[Qnonzero]

        print('\n number of sys  phonon mode =', len(Pnonzero))
        print('\n number of bath phonon mode =', len(Qnonzero))
        print('With SVD mode projection, system modes cm-1 = ', omega_s / cm2mev)
        print('With SVD mode projection, system modes meV = ', omega_s)

        # lines below can be used to write down the outputs in .xsf format
        # fgeo = 'data files/optimized.xyz'
        # fname0 = 'data files/spin_phonon_bare_unscaled.dat'
        # coords, normal_modes = get_modes2(fname0, fgeo, nmode=len(freq))
        # usysTest = np.squeeze(np.array([Ks[0].full(), Ks[1].full(), Ks[2].full()])).transpose()
        # write_normalmode_xsf(coords, normal_modes, range(len(omega_s)), usysTest)


if __name__ == "__main__":
    main()
