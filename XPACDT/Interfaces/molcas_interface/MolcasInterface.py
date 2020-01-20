import numpy as np
import os
import subprocess

from datetime import datetime
from pathlib import Path

from XPACDT.Interfaces.InterfaceTemplate import PotentialInterface
from XPACDT.Interfaces.molcas_interface.Patterns import PATTERNS


class MolcasError(Exception):
    """Error related to MOLCAS."""

    # TODO Write verbose output to file rather than STDERR
    def __init__(self, molcas_output):
        groups = PATTERNS["error"].findall(molcas_output)
        if len(groups) == 0:
            super().__init__(
                f"""MOLCAS error of unknown nature.
                Here is the full MOLCAS output for reference

                {molcas_output}""")

        else:
            section, err = groups[-1]

            super().__init__(
                f"""The following MOLCAS error occured

                {err}

                The full output of the corresponding section is given below

                {section}
                """)


class MolcasInterface(PotentialInterface):
    """
    Electronic structure as computed by the external program MOLCAS.

    The following environnement are (locally) overriden:
    - MOLCAS_PROJECT
    - MOLCAS_WORKDIR, Can be set in the input file
    - MOLCAS_NEW_WORKDIR

    Parameters
    ----------
    max_n_beads : int, optional
        Maximum number of beads from the (n_dof) list of n_beads. Default: 1.

    Input file parameters
    ---------------------
    title : str
        Title of the MOLCAS computation.

    basis : str
        Basis set for the computation (see MOLCAS manual for details).

    charge : int
        Total charge of the system
    """
    # TODO Allow to give the full SEWARD/SCF/RASSCF part of the MOLCAS input
    # file in the XPACDT input file ?
    def __init__(self, max_n_beads=1,
                 workdir=Path.cwd() / "tmp" / "molcas",
                 molcas_executable="molcas",
                 **kwargs):

        if max_n_beads != 1:
            raise NotImplementedError(
                    "Beads not supported for Molcas interface yet.")

        super().__init__(self,
                         "Molcas",
                         1,             # TODO Get n_dof somehow
                         1,             # TODO Determine n_states somehow
                         max_n_beads,
                         **kwargs)

        time_desc = datetime.now().isoformat(timespec="seconds")
        molcas_project_name = f"XPACDT_project_{time_desc}"

        self._workdir = workdir
        self._workdir.mkdir(parents=True, exists_ok=True)

        self._molcas_env = os.environ.copy()

        # Environnements variables used by MOLCAS
        # TODO Allow to give other related env variable in input ?
        # e.g. MOLCAS directory
        self._molcas_env["MOLCAS_PROJECT"] = molcas_project_name
        self._molcas_env["MOLCAS_WORKDIR"] = str(self._workdir)
        self._molcas_env["MOLCAS_NEW_WORKDIR"] = "NO"

        self._molcas_executable = molcas_executable
        self._test_molcas()

        # Wether to reuse result the orbitals of the previous iteration as
        # starting point for the new orbitals computations
        self._compute_from_scratch = True

        self._molcas_runfile = self._workdir / f"{molcas_project_name}.RunFile"
        self._molcas_input_file = self._molcas_runfile.with_suffix("input")

        # Find the template file relative to the current file
        raw_input_template_file = Path(__file__).with_name("template.input")
        raw_input_template = raw_input_template_file.read_text()

        # Partially format the template with the parameters that stay constant
        try:
            self._template_molcas_input = raw_input_template.format(
                title="XPACDT computation",  # TODO Remove or find a more useful title
                xcart="{xcart}",
                )  # TODO pass all other parameters
        except KeyError as err:
            raise ValueError(
                f"Keyword {err.args[0]} must be defined in the MOLCAS "
                "section of the input file. See documentation of the "
                "MolcasInterface class for more information.")

    def _calculate_adiabatic_all(self, R, P, S):
        full_output = self._run_molcas(R)

        energies = PATTERNS["energy"].findall(full_output)
        self._adiabatic_energy = np.array([float(E) for E in energies])

        gradient_section, = PATTERNS["gradient section"].findall(full_output)
        gradients = PATTERNS["gradient"].findall(gradient_section)
        self.adiabatic_gradient = np.stack(
            [np.fromstring(grad, sep=" ") for grad in gradients])

        # TODO NAC directly from Alaska rather than overlap matrix

        overlap_section, = PATTERNS["overlap section"].findall(full_output)
        (header, elements), = PATTERNS["overlap matrix"].findall(overlap_section)
        elements = np.fromstring(elements, sep=" ")

        # If the header has been matched, it means the diagonal form has been
        # used
        if len(header) > 0:
            overlap_matrix = np.diag(elements)
        else:
            n = int((1 + np.sqrt(1 + 4*len(elements)))/2)
            mask_tril = np.tril(np.ones((n, n), dtype=bool))
            mask_triu = np.triu(np.ones((n, n), dtype=bool))

            overlap_matrix = np.zeros(n, n)
            overlap_matrix[mask_tril] = elements
            overlap_matrix[mask_triu] = np.transpose(overlap_matrix)[mask_triu]

    def _molcas_subprocess(self, *args):
        """
        Start molcas as a subprocess and return its output as a string.

        Parameters
        ----------
        All parameters must be strings and are passed as additional parameters
        to the MOLCAS invokation.
        """
        res = subprocess.run(
            [self._molcas_executable, *args],
            env=self._molcas_env,  # Define environnement variables
            text=True,  # Everything treated as string rather than binary
            stdout=subprocess.PIPE)  # Redirect output to the returned object
        # NOTE Binary string could be used (flag text=False), as regex, and
        # float and numpy parsings are compatible with it.

        if res.returncode != 0:
            raise MolcasError(res.stdout)

        return res.stdout

    def _run_molcas(self, R):
        """
        Run MOLCAS for a given configuration.
        """
        input_str = self._template_molcas_input.format(xcart=R)
        self._molcas_input_file.write_text(input_str)

        # NOTE Currently self._compute_from_scratch is always True
        if self._compute_from_scratch:
            self._molcas_env["MOLCAS_NEW_WORKDIR"] = "YES"
        else:
            self._molcas_env["MOLCAS_NEW_WORKDIR"] = "NO"

        return self._molcas_subprocess(self._molcas_input_file)

    def _test_molcas(self):
        """
        Test that the molcas executable name exists and determine its MOLCAS
        version.
        """
        empty_input = Path(__file__).with_name("empty.input")
        try:
            # Run molcas with empty input file to only have base output
            output = self._molcas_subprocess(empty_input)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No MOLCAS executable named {self._molcas_executable} found. "
                "Make sure to add the executable to your PATH or to specify "
                "its full path in the input file.")

        self._molcas_version, = PATTERNS["molcas version"].findall(output)

        print(f"MOLCAS version {self._molcas_version} is used.")

        # TODO check if the provided MOLCAS supports Alaska NAC e.g. using molcas help alaska nac
