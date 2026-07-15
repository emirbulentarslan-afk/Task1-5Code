from pyscf import gto


# ============================================================
# TASK 1: Construct H2, O2 and H2O using experimental geometries
# Coordinates are given in Angstrom.
# Source: NIST CCCBDB
# ============================================================


# -------------------------
# 1. Hydrogen molecule, H2
# Experimental H-H distance: 0.7414 Angstrom
# -------------------------

mol_H2 = gto.M(
    atom="""
    H  0.0000  0.0000  0.0000
    H  0.0000  0.0000  0.7414
    """,
    basis="sto-3g",
    unit="Angstrom",
    charge=0,
    spin=0,
    verbose=0
)


# -------------------------
# 2. Oxygen molecule, O2
# Experimental O-O distance: 1.2075 Angstrom
# O2 is a triplet, therefore spin = 2
# -------------------------

mol_O2 = gto.M(
    atom="""
    O  0.0000  0.0000  0.0000
    O  0.0000  0.0000  1.2075
    """,
    basis="sto-3g",
    unit="Angstrom",
    charge=0,
    spin=2,
    verbose=0
)


# -------------------------
# 3. Water molecule, H2O
# Experimental Cartesian coordinates from NIST CCCBDB
# -------------------------

mol_H2O = gto.M(
    atom="""
    O  0.0000   0.0000   0.1173
    H  0.0000   0.7572  -0.4692
    H  0.0000  -0.7572  -0.4692
    """,
    basis="sto-3g",
    unit="Angstrom",
    charge=0,
    spin=0,
    verbose=0
)


# Print basic information to check that all molecules were built correctly

print("H2 molecule created successfully.")
print("Number of atoms in H2:", mol_H2.natm)
print()

print("O2 molecule created successfully.")
print("Number of atoms in O2:", mol_O2.natm)
print("O2 spin:", mol_O2.spin)
print()

print("H2O molecule created successfully.")
print("Number of atoms in H2O:", mol_H2O.natm)
print()

print("All Task 1 molecules were constructed successfully.")

from pyscf import scf
from pyscf.geomopt.geometric_solver import optimize


# ============================================================
# TASK 2: Geometry optimisation with HF/STO-3G
# ============================================================


# -------------------------
# H2 geometry optimisation
# -------------------------

hf_H2 = scf.RHF(mol_H2)

mol_H2_opt = optimize(hf_H2)

hf_H2_opt = scf.RHF(mol_H2_opt)
energy_H2_opt = hf_H2_opt.kernel()


# -------------------------
# O2 geometry optimisation
# O2 is triplet, so UHF is used
# -------------------------

hf_O2 = scf.UHF(mol_O2)

mol_O2_opt = optimize(hf_O2)

hf_O2_opt = scf.UHF(mol_O2_opt)
energy_O2_opt = hf_O2_opt.kernel()


# -------------------------
# H2O geometry optimisation
# -------------------------

hf_H2O = scf.RHF(mol_H2O)

mol_H2O_opt = optimize(hf_H2O)

hf_H2O_opt = scf.RHF(mol_H2O_opt)
energy_H2O_opt = hf_H2O_opt.kernel()


# -------------------------
# Print optimised coordinates
# -------------------------

print("\nOptimised H2 coordinates:")
print(mol_H2_opt.atom_coords(unit="Angstrom"))

print("\nOptimised O2 coordinates:")
print(mol_O2_opt.atom_coords(unit="Angstrom"))

print("\nOptimised H2O coordinates:")
print(mol_H2O_opt.atom_coords(unit="Angstrom"))


# -------------------------
# Print optimised energies
# -------------------------

print("\nOptimised electronic energies:")

print(f"H2 energy  = {energy_H2_opt:.10f} Hartree")
print(f"O2 energy  = {energy_O2_opt:.10f} Hartree")
print(f"H2O energy = {energy_H2O_opt:.10f} Hartree")


import numpy as np


# ============================================================
# TASK 3: Geometry analysis
# ============================================================


def bond_length(coord1, coord2):
    """
    Calculate the distance between two atoms in Angstrom.
    """
    return np.linalg.norm(coord1 - coord2)


def bond_angle(coord1, center_coord, coord2):
    """
    Calculate the angle coord1-center_coord-coord2 in degrees.
    """
    vector1 = coord1 - center_coord
    vector2 = coord2 - center_coord

    cos_angle = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )

    # Prevent small numerical errors from producing invalid values
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


# Get optimized coordinates in Angstrom

coords_H2 = mol_H2_opt.atom_coords(unit="Angstrom")
coords_O2 = mol_O2_opt.atom_coords(unit="Angstrom")
coords_H2O = mol_H2O_opt.atom_coords(unit="Angstrom")


# -------------------------
# H2 bond length
# -------------------------

H2_bond = bond_length(coords_H2[0], coords_H2[1])


# -------------------------
# O2 bond length
# -------------------------

O2_bond = bond_length(coords_O2[0], coords_O2[1])


# -------------------------
# H2O bond lengths and angle
# Atom order:
# 0 = O
# 1 = H
# 2 = H
# -------------------------

OH1_bond = bond_length(coords_H2O[0], coords_H2O[1])
OH2_bond = bond_length(coords_H2O[0], coords_H2O[2])

HOH_angle = bond_angle(
    coords_H2O[1],
    coords_H2O[0],
    coords_H2O[2]
)


# -------------------------
# Experimental values
# -------------------------

H2_exp = 0.7414
O2_exp = 1.2075
OH_exp = 0.9578
HOH_exp = 104.48


# -------------------------
# Print calculated values
# -------------------------

print("\nTASK 3: OPTIMISED GEOMETRY ANALYSIS")

print(f"\nH2 bond length:")
print(f"Calculated = {H2_bond:.4f} Angstrom")
print(f"Experimental = {H2_exp:.4f} Angstrom")
print(f"Difference = {H2_bond - H2_exp:+.4f} Angstrom")

print(f"\nO2 bond length:")
print(f"Calculated = {O2_bond:.4f} Angstrom")
print(f"Experimental = {O2_exp:.4f} Angstrom")
print(f"Difference = {O2_bond - O2_exp:+.4f} Angstrom")

print(f"\nH2O first O-H bond length:")
print(f"Calculated = {OH1_bond:.4f} Angstrom")
print(f"Experimental = {OH_exp:.4f} Angstrom")
print(f"Difference = {OH1_bond - OH_exp:+.4f} Angstrom")

print(f"\nH2O second O-H bond length:")
print(f"Calculated = {OH2_bond:.4f} Angstrom")
print(f"Experimental = {OH_exp:.4f} Angstrom")
print(f"Difference = {OH2_bond - OH_exp:+.4f} Angstrom")

print(f"\nH-O-H bond angle:")
print(f"Calculated = {HOH_angle:.2f} degrees")
print(f"Experimental = {HOH_exp:.2f} degrees")
print(f"Difference = {HOH_angle - HOH_exp:+.2f} degrees")

# ============================================================
# TASK 4: Reaction energy
# 2 H2O -> 2 H2 + O2
# ============================================================

HARTREE_TO_EV = 27.2114

delta_E_hartree = (
    2 * energy_H2_opt
    + energy_O2_opt
    - 2 * energy_H2O_opt
)

delta_E_eV = delta_E_hartree * HARTREE_TO_EV

print("\nTASK 4: WATER-SPLITTING REACTION ENERGY")

print(f"H2 energy  = {energy_H2_opt:.10f} Hartree")
print(f"O2 energy  = {energy_O2_opt:.10f} Hartree")
print(f"H2O energy = {energy_H2O_opt:.10f} Hartree")

print("\nReaction:")
print("2 H2O -> 2 H2 + O2")

print(f"\nReaction energy = {delta_E_hartree:.10f} Hartree")
print(f"Reaction energy = {delta_E_eV:.4f} eV")

# ============================================================
# TASK 5: Converged Hartree-Fock SCF total energies
# ============================================================

print("\nTASK 5: SCF TOTAL ENERGIES")

print("\nMolecule   Method       Converged   Energy (Hartree)")
print("---------------------------------------------------")

print(
    f"H2         RHF/STO-3G   {str(hf_H2_opt.converged):<9s}   "
    f"{hf_H2_opt.e_tot:.10f}"
)

print(
    f"O2         UHF/STO-3G   {str(hf_O2_opt.converged):<9s}   "
    f"{hf_O2_opt.e_tot:.10f}"
)

print(
    f"H2O        RHF/STO-3G   {str(hf_H2O_opt.converged):<9s}   "
    f"{hf_H2O_opt.e_tot:.10f}"
)
