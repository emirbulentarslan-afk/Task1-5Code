#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 22:01:54 2026

@author: ekinirmak
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 20:25:37 2026

@author: ekinirmak
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:48:20 2026

@author: ekinirmak
"""
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
    f"{hf_H2O_opt.e_tot:.10f}")


basis2= "3-21G"
basis3="6-31G"
basis4="cc-pVDZ"
basis5="cc-pVTZ"
basis6="cc-pVQZ"
basis7="cc-pV5Z"

def geometry_to_atom_string(molecule):
    """
    Convert a PySCF molecule geometry into an atom string
    with coordinates in Angstrom.
    """
    coordinates = molecule.atom_coords(unit="Angstrom")
    atom_lines = []

    for i in range(molecule.natm):
        symbol = molecule.atom_symbol(i)
        x, y, z = coordinates[i]

        atom_lines.append(
            f"{symbol} {x:.10f} {y:.10f} {z:.10f}"
        )

    return "\n".join(atom_lines)

H2_geometry = geometry_to_atom_string(mol_H2_opt)
O2_geometry = geometry_to_atom_string(mol_O2_opt)
H2O_geometry = geometry_to_atom_string(mol_H2O_opt)

from pyscf import gto, scf


def calculate_hf_energy(atom_string, basis, spin):
    """
    Calculate the Hartree-Fock total electronic energy
    for a fixed molecular geometry.
    """

    mol = gto.M(
        atom=atom_string,
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=spin,
        verbose=0
    )

    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    energy = mf.kernel()

    if not mf.converged:
        print(f"Warning: SCF did not converge for basis {basis}")

    return energy, mf.converged

basis_sets = [
    "sto-3g",
    "3-21g",
    "6-31g",
    "cc-pvdz",
    "cc-pvtz",
    "cc-pvqz",
    "cc-pv5z" ]
    
HARTREE_TO_EV = 27.2114

results = []

for basis in basis_sets:

    print(f"Running basis set: {basis}")

    E_H2, conv_H2 = calculate_hf_energy(
        H2_geometry,
        basis,
        spin=0
    )

    E_O2, conv_O2 = calculate_hf_energy(
        O2_geometry,
        basis,
        spin=2
    )

    E_H2O, conv_H2O = calculate_hf_energy(
        H2O_geometry,
        basis,
        spin=0
    )

    delta_E_hartree = (
        2 * E_H2
        + E_O2
        - 2 * E_H2O
    )

    delta_E_eV = delta_E_hartree * HARTREE_TO_EV

    all_converged = conv_H2 and conv_O2 and conv_H2O

    results.append({
        "basis": basis,
        "E_H2": E_H2,
        "E_O2": E_O2,
        "E_H2O": E_H2O,
        "delta_E_hartree": delta_E_hartree,
        "delta_E_eV": delta_E_eV,
        "converged": all_converged
    })
    
    print("\nTASK 6: BASIS SET CONVERGENCE")

print(
    "\nBasis set     E_H2          E_O2           "
    "E_H2O          Delta E (eV)    Converged"
)

print("-" * 85)

for result in results:
    print(
        f"{result['basis']:<12s}"
        f"{result['E_H2']:>14.8f}"
        f"{result['E_O2']:>16.8f}"
        f"{result['E_H2O']:>16.8f}"
        f"{result['delta_E_eV']:>15.6f}"
        f"{str(result['converged']):>12s}"
    )
    
    reference_energy = None

for result in results:
    if result["basis"] == "cc-pvqz":
        reference_energy = result["delta_E_eV"]
        break
    
for result in results:
    result["difference_from_ccpVQZ"] = abs(
        result["delta_E_eV"] - reference_energy
    )
    
    print("\nComparison with cc-pVQZ reference")

print("\nBasis set     Delta E (eV)     Difference from cc-pVQZ (eV)")
print("-" * 60)

for result in results:
    print(
        f"{result['basis']:<12s}"
        f"{result['delta_E_eV']:>15.6f}"
        f"{result['difference_from_ccpVQZ']:>25.6f}"
    )
    
    
import matplotlib.pyplot as plt

basis_labels = [result["basis"] for result in results]
reaction_energies = [result["delta_E_eV"] for result in results]

plt.figure(figsize=(8, 5))

plt.plot(
    basis_labels,
    reaction_energies,
    marker="o"
)

plt.xlabel("Basis set")
plt.ylabel("Reaction energy, ΔEr (eV)")
plt.title("Basis-set convergence of the water-splitting reaction energy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

differences = [
    result["difference_from_ccpVQZ"]
    for result in results
]

plt.figure(figsize=(8, 5))

plt.bar(
    basis_labels,
    differences
)

plt.xlabel("Basis set")
plt.ylabel("Absolute difference from cc-pVQZ (eV)")
plt.title("Deviation from the cc-pVQZ reference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# TASK 7: SINGLE MOLECULE VS REACTION ENERGY CONVERGENCE
# ============================================================
import matplotlib.pyplot as plt
import numpy as np

# Extracting data from the 'results' list in Task 6
basis_labels = [r["basis"] for r in results]
e_h2o_values = [r["E_H2O"] for r in results]        # H2O total energy in Hartree
delta_e_values = [r["delta_E_eV"] for r in results] # Reaction energy in eV

# --- PLOT 1: Dual Y-Axis Absolute Value Plot ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left Axis: H2O Total Energy
color1 = 'tab:blue'
ax1.set_xlabel('Basis Set', fontsize=12)
ax1.set_ylabel('H₂O Total Energy (Hartree)', color=color1, fontsize=12)
line1 = ax1.plot(basis_labels, e_h2o_values, marker='s', color=color1, linewidth=2, label='H₂O Total Energy')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.tick_params(axis='x', rotation=45)

# Right Axis: Reaction Energy
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Reaction Energy $\Delta E_r$ (eV)', color=color2, fontsize=12)
line2 = ax2.plot(basis_labels, delta_e_values, marker='o', color=color2, linewidth=2, label='Reaction Energy')
ax2.tick_params(axis='y', labelcolor=color2)

# Merging legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title("Convergence Comparison: Single Molecule vs Reaction Energy")
plt.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
plt.show()

# --- PLOT 2: Deviation from Reference (Error) Plot (Logarithmic) ---
# Using cc-pV5Z, which gives the best result, as the reference
ref_e_h2o = e_h2o_values[-1]
ref_delta_e = delta_e_values[-1]

# Calculating absolute errors
error_h2o = [abs(e - ref_e_h2o) for e in e_h2o_values]
error_delta_e = [abs(e - ref_delta_e) for e in delta_e_values]

plt.figure(figsize=(10, 6))
# Since the difference of cc-pV5Z with itself is 0, it will cause an error in the log plot, 
# therefore we exclude the last element from the plot using [:-1].
plt.plot(basis_labels[:-1], error_h2o[:-1], marker='s', color='tab:blue', linewidth=2, label='Error in H₂O Energy (Hartree)')
plt.plot(basis_labels[:-1], error_delta_e[:-1], marker='o', color='tab:red', linewidth=2, label='Error in $\Delta E_r$ (eV)')

plt.yscale('log')
plt.xlabel('Basis Set', fontsize=12)
plt.ylabel('Absolute Error relative to cc-pV5Z', fontsize=12)
plt.title("Convergence Speed: Error Decay (Log Scale)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5, which='both')
plt.tight_layout()
plt.show()