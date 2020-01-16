import numpy as np
import os
import sys

from pathlib import Path

from XPACDT.Interfaces.InterfaceTemplate import PotentialInterface
from XPACDT.Interfaces.molcas_interface.Patterns import PATTERNS


class MolcasInterface(PotentialInterface):
    def __init__(self, max_n_beads=1, **kwargs):
        if max_n_beads != 1:
            raise NotImplementedError("Beads not supported for Molcas interface yet.")

        super().__init__(self,
                         "Molcas",
                         1,             # TODO What is that ?
                         1,             # TODO And that ?
                         max_n_beads,
                         **kwargs)

        # Wether to reuse result the orbitals of the previous iteration as 
        # starting point for the new orbitals computations
        self._compute_from_scratch = False

        # TODO Decide system input (WorkDir, etc)
        # TODO validate input
        # TODO creates files and temp dir
        # TODO init filenames and paths

        with relative_open(__file__, "templates/run_molcas.sh") as template:
            self._template_run_molcas = template.read()

        # Partially format the template with the parameter that stay constant
        self._template_run_molcas.format(
                project_name    = name,
                workdir         = workdir,
                input_filename  = input_filename,
                from_scratch    = "{from_scratch}")

        with relative_open(__file__, "templates/molcas.input") as template:
            self._template_molcas_input = template.read()

        # Partially format the template with the parameter that stay constant
        self._template_molcas_input.format(
                title       = xx,
                basis       = xx,
                natoms      = xx,
                charge      = xx,
                spin        = xx,
                nactel      = xx,
                ras2        = xx,
                ciroot      = xx,
                rlxroot     = xx,
                xcart       = "{xcart}"
        )

    def _calculate_adiabatic_all(self, R, P, S):
        # TODO Find a way to determine wether computation must be started from scratch
        self._run_molcas(R)

        with open(self._filepath_molcas_output) as molcas_output:
            full_output = molcas_output.read()
        
        # TODO remove molcas log files (i.e. only keep orbitals file)

        energies = PATTERNS["energy"].findall(full_output)
        self._adiabatic_energy = np.array([float(E) for E in energies])

        gradient_section, = PATTERNS["gradient section"].findall(full_output)

        gradients = PATTERNS["gradient"].findall(gradient_section)
        # TODO check concatenation
        self.adiabatic_gradient = np.vcat(
            [np.fromstring(grad, sep=" ") for grad in gradients]
        )

        overlap_section, = PATTERNS["overlap section"].findall(full_output)

        # TODO Check if good and implement non diagonal case
        diagonal, = PATTERNS["overlap diagonal"].findall(overlap_section)
        overlap_matrix = np.diag(np.fromstring(diagonal, sep=" "))

        #TODO get NAC from overlap matrix


    def _run_molcas(self, R):
        with open(self._filepath_molcas_input, "w") as inputfile:
            input_str = self._template_molcas_input.format(xcart=R)
            inputfile.write(input_str)

        if self._compute_from_scratch:
            from_scratch = "YES"
        else:
            from_scratch = "NO"

        command = self._template_run_molcas.format(from_scratch=from_scratch)
        os.system(command)

def relative_open(filepath, path, *args, **kwargs):
    relative_path = (Path(filepath).parent / path).resolve()
    return open(relative_path, *args, **kwargs)


# TODO take in account all of what is described below
"""
Perform a rasci calculation and return the gradient for one root

Also compute the matrix of time derivative couplings D by
    D_ij = (1/2dt) (<P_i(t)|P_j(t+dt)> - <P_i(t+dt)|P_j(t)>)

Input
    X       --  3N array with geometry in bohr
    V       --  3N array with atomic velocities in bohr/au
    Time    --  Simulation time
    root_g  -- root for which gradient is returned

The symmetric forward 1st order expression

    D_ij = (1/2dt) (<P_i(t)|P_j(t+dt)> - <P_i(t+dt)|P_j(t)>)

is implemented using the nuclear velocities as

    D_ij = (1/2dt) (<P_i(X)|P_j(X+V*dt)> - <P_i(X+V*dt)|P_j(X)>)

where dt is the internal parameter controlling the finite
differentiation step.
        """
"""
Extract the useful matrix elements from the overlap matrix of rassi

The overlap matrix provided by rassi contains useless blocs with overlaps
between eigenstates states at the same geometry.
This The S_ij matrix has elements

<Psi_i(x_a) | Psi_j(x_b)>

where x_a and x_b are different nuclear configurations
"""
# TODO: Adapt that part (previous part is crap from bad usage of MOLCAS)
# Fix phases of S matrix:
# S_ii must be positive
# In case it is negative, the ket wave function has changed phase with
# respect to the corresponding bra. We fix it by multiplying the
# corresponding column by -1