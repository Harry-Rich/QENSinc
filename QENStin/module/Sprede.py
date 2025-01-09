import kinisi.parser as parser
from dynasor.qpoints import get_spherical_qpoints
from typing import List, Tuple, Union
import numpy as np
import pickle
from tqdm import tqdm
import scipp as sc
from scipp.typing import VariableLikeType


class Sprede:
    """
    Base class for incoherent scattering calculations, produces bound incoherent scattering lengths based on chosen isotopes.
    
    :param kinisi_trajectory: Parsed trajectory as a kinisi object
    :param structure: structure as produced by kinisi
    :param specie: Specie to calculate displacements for as a String, e.g. :py:attr:`'Li'`.
    :param isotopes: List of isotopes formatted based on the Neutron scattering lengths and cross sections from NIST https://www.ncnr.nist.gov/resources/n-lengths/list.html
    :param time_step: Time step, in picoseconds, between steps in trajectory.
    :param step_skip: Sampling freqency of the trajectory (time_step is multiplied by this number to get the real
    :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
    
    """

    def __init__(self,
                 kinisi_trajectory,
                 structure: np.ndarray,
                 specie: str,
                 isotopes: List[str],
                 time_step: VariableLikeType,
                 step_skip: VariableLikeType,
                 dimensions: np.ndarray,
                 progress: bool = True):

        total_scat_lengths = np.full(len(structure), np.nan)

        with open('incoherent_scattering_length.pkl', 'rb') as f:
            # bound incoherent scattering length
            b_inco_dict = pickle.load(f)

        for spec in specie:
            ind = kinisi_trajectory.get_indices(structure,
                                                specie=spec,)[0]

            total_scat_lengths[ind.values] = b_inco_dict[isotopes[specie.index(spec)]]

        total_scat_lengths = [
            x for x in total_scat_lengths if np.isnan(x) == False
        ]
        total_scat_lengths = np.array(total_scat_lengths)
        self.total_scat_lengths = total_scat_lengths
        self.kinisi_trajectory = kinisi_trajectory
        self.dimensions = dimensions

    def calculate_q_points(self, q_max, max_points=None, seed=42):
        """
        Produces isotoropically sampled spherical q_points based on the dimensions of the MD cell using Dynasor's get_spherical_qpoints function. 
        For more information see: https://dynasor.materialsmodeling.org/interface_python/q_points.html#dynasor.qpoints.get_spherical_qpoints

        :param q_max: maximum norm of generated q-points
        :param max_points: Optionally limit the set to approximately :py:attr:`max_points` points by randomly removing points from a “fully populated mesh”. 
        The points are removed in such a way that for , the points will be radially uniformly distributed. The value of is calculated from max_q, max_points, and the shape of the cell.
        :param seed: Random seed for stochastic pruning

        """

        q_points = get_spherical_qpoints(cell=self.dimensions,
                                         q_max=q_max,
                                         max_points=max_points)
        return q_points

    def calculate_Finc_qt(self, q_points: np.ndarray, normalise=True):
        """
        Calculate the incoherent intermediete scattering function F(q,t) as per eqn17 in https://doi.org/10.1016/0010-4655(95)00048-K

        :param q_points: Chosen q_points in 3 dimensional cartesian coordinates

        :returns: Numpy array [q_points x time_interval] giving the incoherent intermediate scattering function Finc(q,t)
        
        """
        incoh_f = np.zeros(
            (len(q_points), len(self.kinisi_trajectory.displacements.values)))
        s_len_sq = self.total_scat_lengths**2

    
        for i in tqdm(range(1, len(self.kinisi_trajectory.dt)+1)):

            disp = sc.concat([self.kinisi_trajectory.displacements['obs', i - 1], 
                              self.kinisi_trajectory.displacements['obs', i:] - self.kinisi_trajectory.displacements['obs', :-i]],
                              'obs')
 
            # Probably try and make these variable names have some physical significance
            traj_q_combo = np.einsum('ijk,lk->ijl',
                                     disp.values,
                                     q_points)
       
            expensive_exponent = np.exp(1j * traj_q_combo)
            mean_1 = np.mean(expensive_exponent, axis=0)
            incoh_f[:, i-1] = np.mean(mean_1 * s_len_sq[:, np.newaxis], axis=0)
        
        if normalise == True: 
            incoh_f =  incoh_f / incoh_f[0,0]

        return incoh_f

    @property
    def delta_t(self):
        dt = self.kinisi_trajectory.dt
        return dt
    

    @property
    def scattering_lengths(self):
        scattering_lengths = self.total_scat_lengths
        return scattering_lengths



class MDAnalysisParser(Sprede):
    """
    A parser that consumes an MDAnalysis.Universe object, almost entirely based on the kinisi.parser.MDAnalysisParser but also takes an additional
    isotope list. 

    :param universe: The MDAnalysis object of interest.
    :param specie: Specie to calculate displacements for as a String, e.g. :py:attr:`'Li'`.
    :param isotopes: List of isotopes formatted based on the Neutron scattering lengths and cross sections from NIST https://www.ncnr.nist.gov/resources/n-lengths/list.html
    :param time_step: Time step, in picoseconds, between steps in trajectory.
    :param step_skip: Sampling freqency of the trajectory (time_step is multiplied by this number to get the real
        time between output from the simulation file).
    :param sub_sample_atoms: The sampling rate to sample the atoms in the system. Optional, defaults
        to :py:attr:`1` where all atoms are used.
    :param sub_sample_traj: Multiple of the :py:attr:`time_step` to sub sample at. Optional,
        defaults to :py:attr:`1` where all timesteps are used.
    :param min_dt: Minimum time interval to be evaluated, in the simulation units. Optional, defaults to the
        produce of :py:attr:`time_step` and :py:attr:`step_skip`.
    :param max_dt: Maximum time interval to be evaluated, in the simulation units. Optional, defaults to the
        length of the simulation.
    :param n_steps: Number of steps to be used in the time interval function. Optional, defaults to :py:attr:`100`
        unless this is fewer than the total number of steps in the trajectory when it defaults to this number.
    :param spacing: The spacing of the steps that define the time interval, can be either :py:attr:`'linear'` or
        :py:attr:`'logarithmic'`. If :py:attr:`'logarithmic'` the number of steps will be less than or equal
        to that in the :py:attr:`n_steps` argument, such that all values are unique. Optional, defaults to
        :py:attr:`linear`.
    :param sampling: The ways that the time-windows are sampled. The options are :py:attr:`'single-origin'`
        or :py:attr:`'multi-origin'` with the former resulting in only one observation per atom per
        time-window and the latter giving the maximum number of origins without sampling overlapping
        trajectories. Optional, defaults to :py:attr:`'multi-origin'`.
    :param memory_limit: Upper limit in the amount of computer memory that the displacements can occupy in
        gigabytes (GB). Optional, defaults to :py:attr:`8.`.
    :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
    :param specie_indices: Optional, list of indices to calculate diffusivity for as a list of indices. Specie 
        must be set to None for this to function. Molecules can be specificed as a list of lists of indices.
        The inner lists must all be on the same length.
    :param masses: Optional, list of masses associated with the indices in specie_indices. Must be same shape as specie_indices.
    :param framework_indices: Optional, list of framework indices to be used to correct framework drift. If an empty list is passed no drift correction will be performed.
    """

    def __init__(self,
                 universe: 'MDAnalysis.core.universe.Universe',
                 specie: str,
                 isotopes: List[str],
                 time_step: VariableLikeType,
                 step_skip: VariableLikeType,
                 dt: VariableLikeType = None,
                 dimension: str = 'xyz',
                 distance_unit: sc.Unit = sc.units.angstrom,
                 specie_indices: VariableLikeType = None,
                 masses: VariableLikeType = None,
                 progress: bool = True):

        kinisi_trajectory = parser.MDAnalysisParser(
            universe,
            specie=specie,
            time_step=time_step,
            step_skip=step_skip,
            specie_indices=specie_indices,
            masses=masses,
            progress=progress,)

        # Can remove this call if I can access the structure from the kinisi_trajectory object
        structure, coords, latt = kinisi_trajectory.get_structure_coords_latt(universe = universe)
        dimensions = np.diag(structure.dimensions[0:3])

        super().__init__(kinisi_trajectory=kinisi_trajectory,
                         structure=structure,
                         specie=specie,
                         isotopes=isotopes,
                         time_step=time_step,
                         step_skip=step_skip,
                         dimensions = dimensions,
                         progress=progress)

class PymatgenParser(Sprede):
    """
    A parser for pymatgen structures.

    :param structures: Structures ordered in sequence of run.
    :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
    :param isotopes: List of isotopes formatted based on the Neutron scattering lengths and cross sections from NIST https://www.ncnr.nist.gov/resources/n-lengths/list.html
    :param time_step: Time step, in picoseconds, between steps in trajectory.
    :param step_skip: Sampling freqency of the trajectory (time_step is multiplied by this number to get the real
        time between output from the simulation file).
    :param sub_sample_traj: Multiple of the :py:attr:`time_step` to sub sample at. Optional, defaults
        to :py:attr:`1` where all timesteps are used.
    :param min_dt: Minimum time interval to be evaluated, in the simulation units. Optional, defaults to the
        produce of :py:attr:`time_step` and :py:attr:`step_skip`.
    :param max_dt: Maximum time interval to be evaluated, in the simulation units. Optional, defaults to the
        length of the simulation.
    :param n_steps: Number of steps to be used in the time interval function. Optional, defaults to :py:attr:`100`
        unless this is fewer than the total number of steps in the trajectory when it defaults to this number.
    :param spacing: The spacing of the steps that define the time interval, can be either :py:attr:`'linear'` or
        :py:attr:`'logarithmic'`. If :py:attr:`'logarithmic'` the number of steps will be less than or equal
        to that in the :py:attr:`n_steps` argument, such that all values are unique. Optional, defaults to
        :py:attr:`linear`.
    :param sampling: The ways that the time-windows are sampled. The options are :py:attr:`'single-origin'`
        or :py:attr:`'multi-origin'` with the former resulting in only one observation per atom per
        time-window and the latter giving the maximum number of origins without sampling overlapping
        trajectories. Optional, defaults to :py:attr:`'multi-origin'`.
    :param memory_limit: Upper limit in the amount of computer memory that the displacements can occupy in
        gigabytes (GB). Optional, defaults to :py:attr:`8.`.
    :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
    :param specie_indices: Optional, list of indices to calculate diffusivity for as a list of indices. Specie 
        must be set to None for this to function. Molecules can be specificed as a list of lists of indices.
        The inner lists must all be on the same length.
    :param masses: Optional, list of masses associated with the indices in specie_indices. Must be same shape as specie_indices.
    :param framework_indices: Optional, list of framework indices to be used to correct framework drift. If an empty list is passed no drift correction will be performed.
    """

    def __init__(self,
                 structures: List['pymatgen.core.structure.Structure'],
                 specie: Union['pymatgen.core.periodic_table.Element', 'pymatgen.core.periodic_table.Specie'],
                 isotopes: List[str],
                 time_step: VariableLikeType,
                 step_skip: VariableLikeType,
                 dt: VariableLikeType = None,
                 dimension: str = 'xyz',
                 distance_unit: sc.Unit = sc.units.angstrom,
                 specie_indices: VariableLikeType = None,
                 masses: VariableLikeType = None,
                 progress: bool = True,):
                
        kinisi_trajectory = parser.PymatgenParser(
            structures = structures,
            specie=specie,
            time_step=time_step,
            step_skip=step_skip,
            dt = dt,
            distance_unit = distance_unit,
            specie_indices=specie_indices,
            masses=masses,
            progress=progress)
        
        

        # This call is because of https://github.com/bjmorgan/kinisi/issues/88 - will remove when issue is fixed
        structure, coords, latt = kinisi_trajectory.get_structure_coords_latt(structures=structures, progress=progress)
        kinisi_trajectory.displacements = kinisi_trajectory.calculate_displacements(coords = coords, lattice= latt)

        dimensions = latt.values[0]

        super().__init__(kinisi_trajectory=kinisi_trajectory,
                         structure=structure,
                         specie=specie,
                         isotopes=isotopes,
                         time_step=time_step,
                         step_skip=step_skip,
                         dimensions = dimensions,
                         progress=progress)