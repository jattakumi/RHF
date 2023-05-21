import math
import numpy as np
from scipy import linalg
from scipy import special


class primitive_gaussian:
    def __init__(self, alpha, coeff, coordinates, l1, l2, l3):
        self.l2 = l2
        self.l1 = l1
        self.l3 = l3
        self.alpha = alpha
        self.coeff = coeff
        self.coordinates = np.array(coordinates)
        self.A = (2.0 * alpha / math.pi) ** 0.75


# A function defining the overlap integral
def overlap(molecule):
    nbasis = len(molecule)
    S = np.zeros([nbasis, nbasis])
    for i in range(nbasis):
        for j in range(nbasis):
            nprimitive_i = len(molecule[i])
            nprimitive_j = len(molecule[j])

            for k in range(nprimitive_i):
                for l in range(nprimitive_j):
                    N = molecule[i][k].A * molecule[j][l].A
                    p = molecule[i][k].alpha + molecule[j][l].alpha
                    q = molecule[i][k].alpha * molecule[j][l].alpha / p
                    Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                    Q2 = np.dot(Q, Q)
                    S[i, j] += N * molecule[i][k].coeff * molecule[j][l].coeff * math.exp(-q * Q2) * (math.pi / p) ** (
                        1.5)
    return S


# A function defining the kinetic integral
def kinetic(molecule):
    nbasis = len(molecule)
    T = np.zeros([nbasis, nbasis])
    for i in range(nbasis):
        for j in range(nbasis):
            nprimitive_i = len(molecule[i])
            nprimitive_j = len(molecule[j])

            for k in range(nprimitive_i):
                for l in range(nprimitive_j):
                    cacb = molecule[i][k].coeff * molecule[j][l].coeff
                    N = molecule[i][k].A * molecule[j][l].A
                    p = molecule[i][k].alpha + molecule[j][l].alpha
                    q = molecule[i][k].alpha * molecule[j][l].alpha / p
                    Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                    Q2 = np.dot(Q, Q)
                    P = molecule[i][k].alpha * molecule[i][k].coordinates + molecule[j][l].alpha * molecule[j][
                        l].coordinates
                    Pp = P / p
                    PG = Pp - molecule[j][l].coordinates
                    PGx2 = PG[0] * PG[0]
                    PGy2 = PG[1] * PG[1]
                    PGz2 = PG[2] * PG[2]
                    s = N * cacb * math.exp(-q * Q2) * (math.pi / p) ** (3 / 2)
                    T[i, j] += 3.0 * molecule[j][l].alpha * s
                    T[i, j] -= 2.0 * molecule[j][l].alpha * molecule[j][l].alpha * s * (PGx2 + 0.5 / p)
                    T[i, j] -= 2.0 * molecule[j][l].alpha * molecule[j][l].alpha * s * (PGy2 + 0.5 / p)
                    T[i, j] -= 2.0 * molecule[j][l].alpha * molecule[j][l].alpha * s * (PGz2 + 0.5 / p)
    return T


# Defining the Boys function
def boys(x, n):
    if x == 0:
        return 1.0 / (2 * n + 1)
    else:
        return special.gammainc(n + 0.5, x) * special.gamma(n + 0.5) * (1.0 / (2 *
                                                                               x ** (n + 0.5)))


# Defining the electron-nuclear attraction integral
def electron_nuclear_attraction(molecule, Z):
    natoms = len(Z)
    nbasis = len(molecule)

    coordinates = []
    for i in range(nbasis):
        nprimitive_i = len(molecule[i])
        for j in range(nprimitive_i):
            coordinates.append(molecule[i][j].coordinates)
    coordinates = np.array(coordinates)
    coordinates = np.unique(coordinates, axis=0)
    V_ne = np.zeros([nbasis, nbasis])

    for atom in range(natoms):
        for i in range(nbasis):
            for j in range(nbasis):
                nprimitive_i = len(molecule[i])
                nprimitive_j = len(molecule[j])

                for k in range(nprimitive_i):
                    for l in range(nprimitive_j):
                        N = molecule[i][k].A * molecule[j][l].A
                        cacb = molecule[i][k].coeff * molecule[j][l].coeff
                        p = molecule[i][k].alpha + molecule[j][l].alpha
                        q = molecule[i][k].alpha * molecule[j][l].alpha / p
                        Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                        Q2 = np.dot(Q, Q)
                        P = molecule[i][k].alpha * molecule[i][k].coordinates + molecule[j][l].alpha * molecule[j][
                            l].coordinates
                        Pp = P / p
                        PG = Pp - coordinates[atom]
                        PG2 = np.dot(PG, PG)
                        V_ne[i, j] += -Z[atom] * N * cacb * math.exp(-q * Q2) * (2.0 * math.pi / p) * boys(p * PG2, 0)
    return V_ne


# Describing the electron_electron_repulsion integral of the H2 molecule
def electron_electron_repulsion(molecule):
    nbasis = len(molecule)
    V_ee = np.zeros([nbasis, nbasis, nbasis, nbasis])
    for i in range(nbasis):
        for j in range(nbasis):
            for k in range(nbasis):
                for l in range(nbasis):
                    nprimitive_i = len(molecule[i])
                    nprimitive_j = len(molecule[j])
                    nprimitive_k = len(molecule[k])
                    nprimitive_l = len(molecule[l])

                    for ii in range(nprimitive_i):
                        for jj in range(nprimitive_j):
                            for kk in range(nprimitive_k):
                                for ll in range(nprimitive_l):
                                    N = molecule[i][ii].A * molecule[j][jj].A * molecule[k][kk].A * molecule[l][ll].A
                                    cicjckcl = molecule[i][ii].coeff * molecule[j][jj].coeff * molecule[k][kk].coeff * \
                                               molecule[l][ll].coeff
                                    pij = molecule[i][ii].alpha + molecule[j][jj].alpha
                                    pkl = molecule[k][kk].alpha + molecule[l][ll].alpha
                                    Pij = molecule[i][ii].alpha * molecule[i][ii].coordinates + molecule[j][jj].alpha * \
                                          molecule[j][jj].coordinates
                                    Pkl = molecule[k][kk].alpha * molecule[k][kk].coordinates + molecule[l][ll].alpha * \
                                          molecule[l][ll].coordinates
                                    Ppij = Pij / pij
                                    Ppkl = Pkl / pkl
                                    PpijPpkl = Ppij - Ppkl
                                    PpijPpkl2 = np.dot(PpijPpkl, PpijPpkl)
                                    denom = 1.0 / pij + 1.0 / pkl
                                    qij = molecule[i][ii].alpha * molecule[j][jj].alpha / pij
                                    qkl = molecule[k][kk].alpha * molecule[l][ll].alpha / pkl
                                    Qij = molecule[i][ii].coordinates - molecule[j][jj].coordinates
                                    Qkl = molecule[k][kk].coordinates - molecule[l][ll].coordinates
                                    Q2ij = np.dot(Qij, Qij)
                                    Q2kl = np.dot(Qkl, Qkl)
                                    term1 = 2.0 * math.pi * math.pi / (pij * pkl)
                                    term2 = math.sqrt(math.pi / (pij + pkl))
                                    term3 = math.exp(-qij * Q2ij)
                                    term4 = math.exp(-qkl * Q2kl)
                                    V_ee[i, j, k, l] += N * cicjckcl * term1 * term2 * term3 * term4 * boys(
                                        PpijPpkl2 / denom, 0)
    return V_ee


# Defining the nuclear-nuclear repulsion integral
def nuclear_nuclear_repulsion_energy(atoms_coords, Z):
    assert (len(atoms_coords)) == len(Z)
    natoms = len(Z)
    E_NN = 0
    for i in range(natoms):
        Zi = Z[i]
        for j in range(natoms):
            if j > i:
                Zj = Z[j]

                Rijx = atoms_coords[i][0] - atoms_coords[j][0]
                Rijy = atoms_coords[i][1] - atoms_coords[j][1]
                Rijz = atoms_coords[i][2] - atoms_coords[j][2]

                Rijx_squared = Rijx * Rijx
                Rijy_squared = Rijy * Rijy
                Rijz_squared = Rijz * Rijz

                Rij = math.sqrt(Rijx_squared + Rijy_squared + Rijz_squared)
                E_NN += (Zi * Zj) / Rij
    return E_NN


def compute_G(density_matrix, Vee):
    # nbasis_functions = density_matrix.shape[0]
    # G = np.zeros((nbasis_functions, nbasis_functions))
    G = np.einsum("rs,pqrs->pq", density_matrix, Vee - 0.5 * Vee.transpose(0, 3, 2, 1))
    return G


def compute_electronic_energy_expectation_value(density_matrix, T, Vne, G):
    nbasis_functions = T.shape[0]
    Hcore = T + Vne
    Fock = T + Vne + G
    electronic_energy = 0.0
    ''''for i in range(nbasis_functions):
        for j in range(nbasis_functions):
            electronic_energy += density_matrix[i, j] * (Hcore[i, j] + 0.5 * G[i,j])'''''
    electronic_energy = np.einsum("ij,ij", density_matrix, Hcore + 0.5 * G)  # G * 0.5
    return electronic_energy


# A function to compute the density matrix
def compute_density_matrix(mos, number_occupied_orbitals):
    nbasis_functions = mos.shape[0]
    density_matrix = np.zeros((nbasis_functions, nbasis_functions))
    occupation = 2.0
    for i in range(nbasis_functions):
        for j in range(nbasis_functions):
            for oo in range(number_occupied_orbitals):
                C = mos[i, oo]
                C_dagger = mos[j, oo]
                density_matrix[i, j] += C * C_dagger * occupation
    # density_matrix = np.einsum("ij,kj->ik", mos[:, : number_occupied_orbitals],mos[:, :number_occupied_orbitals])
    return density_matrix


class Basis_set:
    def __init__(self, exponents, coefficients, xyz):
        self.exponents = exponents
        self.coefficients = coefficients
        self.xyz = xyz


distances = [round(i * 0.1, 3) for i in range(3, 50)]  # in unit = bohr (a.u of position)
molecule_coordinates = [[[0.0, 0.0, distance / 2.0], [0.0, 0.0, -distance / 2.0]] for distance in distances]
total_energies = []
# 3-21G basis for 1s orbital for a hydrogen
e1 = 0.5447178000E+01  # exponent value
c1 = 0.1562849787E+00  # transformation coefficient
e2 = 0.8245472400E+00  # exponent value
c2 = 0.9046908767E+00  # transformation coefficient
e3 = 0.1831915800E+00  # exponent value
c3 = 1.0000000  # transformation coefficient

molecule_coordinates[0][0] = [0.0, 0.0, 1]  # one of the atom is assumed to be in an origin and the other one displaced
molecule_coordinates[0][1] = [0.0, 0.0, 0]  # either in the x, y or z coordinates
# defining objects for the primitive gaussians and ....
H1_pg1a = primitive_gaussian(e1, c1, molecule_coordinates[0][0], 0, 0, 0)
H1_pg1b = primitive_gaussian(e2, c2, molecule_coordinates[0][0], 0, 0, 0)
H1_pg1c = primitive_gaussian(e3, c3, molecule_coordinates[0][0], 0, 0, 0)
H2_pg1a = primitive_gaussian(e1, c1, molecule_coordinates[0][1], 0, 0, 0)
H2_pg1b = primitive_gaussian(e2, c2, molecule_coordinates[0][1], 0, 0, 0)
H2_pg1c = primitive_gaussian(e3, c3, molecule_coordinates[0][1], 0, 0, 0)

H1_1s = [H1_pg1a, H1_pg1b]
H1_2s = [H1_pg1c]
H2_1s = [H2_pg1a, H2_pg1b]
H2_2s = [H2_pg1c]

number_occupied_orbitals = 1
Z = [1.0, 1.0]
atoms_coords = [np.array(molecule_coordinates[0][0]),
                np.array(molecule_coordinates[0][1])]

# Defining the molecule as an object
molecule = [H1_1s, H1_2s, H2_1s, H2_2s]
# Print the integrals for the overlap, kinetic, electron-electron repulsion, electron-nuclear attraction and nuclear-nuclear repulsion
print("\nOverlap Integral for H2 molecule 3-21G:\n", overlap(molecule))
print("\nKinetic Integral for H2 molecule 3-21G:\n", kinetic(molecule))
print("\nElectron Nuclear Integral for H2 molecule 3-21G:\n", electron_nuclear_attraction(molecule, Z))
print("\nElectron Electron Repulsion Integral for H2 molecule 3-21G:\n", electron_electron_repulsion(molecule))
print("\nNuclear Nuclear Repulsion Integral for H2 molecule 3-21G:\n",
      nuclear_nuclear_repulsion_energy(atoms_coords, Z))


def scf_cycle(molecular_terms, scf_parameters, molecule):
    S, T, Vne, Vee = molecular_terms
    tolerance, max_scf_steps = scf_parameters
    nbasis_functions = len(molecule)
    density_matrix = np.zeros((nbasis_functions, nbasis_functions))
    electronic_energy_old = 0.0
    print("starting scf procedure")
    print(f" iter no. energy delta_e")
    for scf_step in range(max_scf_steps):
        G = compute_G(density_matrix, Vee)
        F = T + Vne + G
        eps, mos = linalg.eigh(F, S)
        density_matrix = compute_density_matrix(mos, number_occupied_orbitals)
        electronic_energy = compute_electronic_energy_expectation_value(density_matrix, T, Vne, G)
        delta_e = abs(electronic_energy - electronic_energy_old)
        electronic_energy_old = electronic_energy
        print(f" {scf_step} {electronic_energy} {delta_e}")
        if delta_e < tolerance:
            break
        if scf_step > 30:
            print("did not converge")
    print(f"Converged in {scf_step} steps and the energy is {electronic_energy}")
    return electronic_energy


def total_energy(electronic_energy, Enn):
    total_energy = electronic_energy + Enn
    return total_energy

S = overlap(molecule)
T = kinetic(molecule)
Vne = electron_nuclear_attraction(molecule, Z)
Vee = electron_electron_repulsion(molecule)
Enn = nuclear_nuclear_repulsion_energy(atoms_coords, Z)
molecular_terms = S, T, Vne, Vee
scf_parameters = [1e-5, 30]
electronic_energy = scf_cycle(molecular_terms, scf_parameters, molecule)
total_energy(electronic_energy, Enn)
print("\nHartree Fock Energy:", total_energy(electronic_energy, Enn))
print("\n")
total_energies.append(total_energy)

def hartree_fock_energy(R):

    molecule_coordinates[0][0] = [0.0, 0.0, 0]
    molecule_coordinates[0][1] = [0.0, 0.0, R]
    # print(molecule_coordinates[0][0])
    H1_pg1a = primitive_gaussian(e1, c1, molecule_coordinates[0][0], 0, 0, 0)
    H1_pg1b = primitive_gaussian(e2, c2, molecule_coordinates[0][0], 0, 0, 0)
    H1_pg1c = primitive_gaussian(e3, c3, molecule_coordinates[0][0], 0, 0, 0)
    H2_pg1a = primitive_gaussian(e1, c1, molecule_coordinates[0][1], 0, 0, 0)
    H2_pg1b = primitive_gaussian(e2, c2, molecule_coordinates[0][1], 0, 0, 0)
    H2_pg1c = primitive_gaussian(e3, c3, molecule_coordinates[0][1], 0, 0, 0)

    H1_1s = [H1_pg1a, H1_pg1b]
    H1_2s = [H1_pg1c]
    H2_1s = [H2_pg1a, H2_pg1b]
    H2_2s = [H2_pg1c]

    number_occupied_orbitals = 1
    Z = [1.0, 1.0]
    atoms_coords = [np.array(molecule_coordinates[0][0]), np.array(molecule_coordinates[0][1])]

    # Defining the molecule as an object
    molecule = [H1_1s, H1_2s, H2_1s, H2_2s]
    S = overlap(molecule)
    T = kinetic(molecule)
    Vne = electron_nuclear_attraction(molecule, Z)
    Vee = electron_electron_repulsion(molecule)
    Enn = nuclear_nuclear_repulsion_energy(atoms_coords, Z)
    molecular_terms = S, T, Vne, Vee
    electronic_energy = scf_cycle(molecular_terms, scf_parameters, molecule)
    hartree = total_energy(electronic_energy, Enn)
    return hartree

# If the absolute magnitude of the first derivative is greater than 10^-6 a.u.,take a Newton-Raphson step
R = 1.0
geom_opt = 0
delta = 0.001
tolerance = 1e-6
while True:
    E = hartree_fock_energy(R)
    R_pos = R + delta
    R_neg = R - delta
    E_prime1 = hartree_fock_energy(R_pos)
    E_prime2 = hartree_fock_energy(R_neg)
    E_prime = (hartree_fock_energy(R_pos) - hartree_fock_energy(R_neg)) / (2 * delta)
    E_double_prime = (hartree_fock_energy(R_pos) + hartree_fock_energy(R_neg) - 2 * hartree_fock_energy(R)) / (delta ** 2)
    derivative = E_prime
    second_derivative = E_double_prime
    R -= (derivative / second_derivative)
    geom_opt += 1
    if abs(derivative) <= tolerance:
        final_energy = hartree_fock_energy(R)
        print("Optimization complete!")
        print("Bond distance: {:.6f} a.u.".format(R))
        print("Geometry optimization energy: {:.6f} a.u.".format(final_energy))
        print("Geom opt iter:", geom_opt)
        break
    # Report the Harmonic vibrational frequency and the bond dissociation energy
    delta2 = delta ** 2
    force_constant = (hartree_fock_energy(R + delta) + hartree_fock_energy(R - delta) - 2 * hartree_fock_energy(R)) / delta ** 2
    mass_H = 1.00784  # Mass of hydrogen atom in atomic mass units (AMU)
    reduced_mass = (mass_H * mass_H) / (mass_H + mass_H)  # Reduced mass of H2 molecule
    harmonic_freq = math.sqrt(force_constant / reduced_mass) / (2 * math.pi * 3e10)  # Convert force constant to frequency
    print("Harmonic Vibrational Frequency:", harmonic_freq, "Hz")
    energy_H = -0.5  # Energy of a hydrogen atom in atomic units (a.u.)
    dissociation_energy = E - 2 * energy_H
    print("Bond Dissociation Energy:", dissociation_energy, "a.u.")
