import numpy as np
from scipy import constants

def electric_dipole_emission(ang_freq: float, polarization: float,
                                rel_permitivitty:float=1.0)-> float:
    """
    calculate the time-averaged power emitted from an oscillating electric field in a homogenous medium
    
    Parameters
    ----------
    ang_freq: float (rad/s)
      - the (real valued) angular frequency at which to evaluate the fields
    polarization: float (Cm)
      - the magnitude of the polarization vector of the dipole
    rel_permitivitty: float
      - the relative permitivitty of the background medium

    Returns
    -------
    dipole_emission: float (W)

    """    
    
    dipole_emission = constants.mu_0*polarization**2*ang_freq**4 / (12*np.pi*constants.c)
    
    return dipole_emission

def electric_dipole_strength(ang_freq: float, dipole_emission: float,
                                rel_permitivitty:float=1.0)-> float:
    """
    calculate the time-averaged power emitted from an oscillating electric field in a homogenous medium
    
    Parameters
    ----------
    ang_freq: float (rad/s)
      - the (real valued) angular frequency at which to evaluate the fields
    emission: float (W)
      - the power emitted by the dipole
    rel_permitivitty: float
      - the relative permitivitty of the background medium

    Returns
    -------
    polarization: float (Cm)

    """    
    polarization = np.sqrt(dipole_emission*12*np.pi*constants.c/(constants.mu_0*ang_freq**4))
    
    return polarization

def electric_dipole_efield_full(ang_freq: float, polarization: np.ndarray,
                                dipole_pos: np.ndarray,
                                eval_pos:np.ndarray, 
                                rel_permitivitty:float=1.0,
                                time:float=0.)-> np.ndarray:
    """
    calculate the full electric field from an oscillating electric dipole in a homogenous medium
    
    Parameters
    ----------
    ang_freq: float (rad/s)
      - the (real valued) angular frequency at which to evaluate the fields
    polarization: (1, 3) np.ndarray dtype: np.float64 (Cm)
      - the polarization vector of the dipole
    dipole_pos: (1, 3) np.ndarray dtype: np.float64 (m)
      - the position of the dipole
    eval_pos: (N, M) np.ndarray dtype: np.float64 (m)
      - the evaluation positions
    rel_permitivitty: float
      - the relative permitivitty of the background medium
    time: float (s)
      - the time at which to evaluate the fields

    Returns
    -------
    e_field: (N, M, 3) np.ndarray dtype np.complex128 (V/m)

    """    
    
    prefactor = 1/(4*np.pi*constants.epsilon_0*rel_permitivitty)
    n_pos = eval_pos.shape[0] # nuber of evaluation positions
    dipole_pos_array = np.tile(dipole_pos, n_pos).reshape(n_pos, 3)
    r = eval_pos-dipole_pos_array # relative distance to dipole
    
    r_abs = np.linalg.norm(r, axis=1)
    r_abs_array = np.tile(r_abs, 3).reshape(3, n_pos).T
    unit_position = r/r_abs_array
    
    phase_factor_positional = np.exp(1j*ang_freq*r_abs_array/constants.c)
    phase_factor_temporal = np.exp(-1j*ang_freq*time)
    phase_factor = phase_factor_positional*phase_factor_temporal
    
    cross_term = np.cross(np.cross(unit_position, polarization), unit_position)
    dot_term = np.dot(unit_position, polarization)
    dot_term_array = np.vstack([dot_term, dot_term, dot_term]).T
    dot_term_full = 3*unit_position*dot_term_array-polarization
    
    far_field = (ang_freq**2/(constants.c**2*r_abs_array))*cross_term
    near_field_term = dot_term_full/r_abs_array**3
    intermediate_field_term = -1j*ang_freq*dot_term_full/(constants.c*r_abs_array**2)
    e_field = prefactor*(far_field+near_field_term+ intermediate_field_term)*phase_factor
    return e_field


def electric_dipole_bfield_full(ang_freq: float, polarization: np.ndarray,
                                dipole_pos: np.ndarray,
                                eval_pos:np.ndarray, 
                                rel_permitivitty:float=1.0,
                                time:float=0.)-> np.ndarray:
    """
    calculate the full magnetric field from an oscillating electric dipole in a homogenous medium
    
    Parameters
    ----------
    ang_freq: float (rad/s)
      - the (real valued) angular frequency at which to evaluate the fields
    polarization: (1, 3) np.ndarray dtype: np.float64 (Cm)
      - the polarization vector of the dipole
    dipole_pos: (1, 3) np.ndarray dtype: np.float64 (m)
      - the position of the dipole
    eval_pos: (N, M) np.ndarray dtype: np.float64 (m)
      - the evaluation positions
    rel_permitivitty: float
      - the relative permitivitty of the background medium
    time: float (s)
      - the time at which to evaluate the fields

    Returns
    -------
    b_field: (N, M, 3) np.ndarray dtype np.complex128 (A/m)

    """    
    
    prefactor = ang_freq**2/(4*np.pi*constants.epsilon_0*rel_permitivitty*constants.c**3)
    n_pos = eval_pos.shape[0] # nuber of evaluation positions
    dipole_pos_array = np.tile(dipole_pos, n_pos).reshape(n_pos, 3)
    r = eval_pos-dipole_pos_array # relative distance to dipole
    
    r_abs = np.linalg.norm(r, axis=1)
    r_abs_array = np.tile(r_abs, 3).reshape(3, n_pos).T
    unit_position = r/r_abs_array
    
    phase_factor_positional = np.exp(1j*ang_freq*r_abs_array/constants.c)
    phase_factor_temporal = np.exp(-1j*ang_freq*time)
    phase_factor = phase_factor_positional*phase_factor_temporal
    
    cross_term = np.cross(unit_position, polarization)
    
    far_field = (1/r_abs_array)*cross_term
    intermediate_field_term = -constants.c*cross_term/(1j*ang_freq*r_abs_array**2)
    b_field = prefactor*(far_field + intermediate_field_term)*phase_factor
    
    return b_field




def electric_dipole_efield_far(ang_freq: float, polarization: np.ndarray,
                                dipole_pos: np.ndarray,
                                eval_pos:np.ndarray, 
                                rel_permitivitty:float=1.0,
                                time:float=0.)-> np.ndarray:
    """
    calculate the full electric field from an oscillating electric dipole in a homogenous medium
    
    Parameters
    ----------
    ang_freq: float (rad/s)
      - the (real valued) angular frequency at which to evaluate the fields
    polarization: (1, 3) np.ndarray dtype: np.float64 (Cm)
      - the polarization vector of the dipole
    dipole_pos: (1, 3) np.ndarray dtype: np.float64 (m)
      - the position of the dipole
    eval_pos: (N, M) np.ndarray dtype: np.float64 (m)
      - the evaluation positions
    rel_permitivitty: float
      - the relative permitivitty of the background medium
    time: float (s)
      - the time at which to evaluate the fields

    Returns
    -------
    e_field: (N, M, 3) np.ndarray dtype np.complex128 (V/m)

    """    
    
    prefactor = 1/(4*np.pi*constants.epsilon_0*rel_permitivitty)
    n_pos = eval_pos.shape[0] # nuber of evaluation positions
    dipole_pos_array = np.tile(dipole_pos, n_pos).reshape(n_pos, 3)
    r = eval_pos-dipole_pos_array # relative distance to dipole
    
    r_abs = np.linalg.norm(r, axis=1)
    r_abs_array = np.tile(r_abs, 3).reshape(3, n_pos).T
    unit_position = r/r_abs_array
    
    phase_factor_positional = np.exp(1j*ang_freq*r_abs_array/constants.c)
    phase_factor_temporal = np.exp(-1j*ang_freq*time)
    phase_factor = phase_factor_positional*phase_factor_temporal
    
    cross_term = np.cross(np.cross(unit_position, polarization), unit_position)
    far_field = (ang_freq**2/(constants.c**2*r_abs_array))*cross_term
    e_field = prefactor*(far_field)*phase_factor
    return e_field


def electric_dipole_bfield_far(ang_freq: float, polarization: np.ndarray,
                                dipole_pos: np.ndarray,
                                eval_pos:np.ndarray, 
                                rel_permitivitty:float=1.0,
                                time:float=0.)-> np.ndarray:
    """
    calculate the full magnetric field from an oscillating electric dipole in a homogenous medium
    
    Parameters
    ----------
    ang_freq: float (rad/s)
      - the (real valued) angular frequency at which to evaluate the fields
    polarization: (1, 3) np.ndarray dtype: np.float64 (Cm)
      - the polarization vector of the dipole
    dipole_pos: (1, 3) np.ndarray dtype: np.float64 (m)
      - the position of the dipole
    eval_pos: (N, M) np.ndarray dtype: np.float64 (m)
      - the evaluation positions
    rel_permitivitty: float
      - the relative permitivitty of the background medium
    time: float (s)
      - the time at which to evaluate the fields

    Returns
    -------
    b_field: (N, M, 3) np.ndarray dtype np.complex128 (A/m)

    """    
    
    prefactor = ang_freq**2/(4*np.pi*constants.epsilon_0*rel_permitivitty*constants.c**3)
    n_pos = eval_pos.shape[0] # nuber of evaluation positions
    dipole_pos_array = np.tile(dipole_pos, n_pos).reshape(n_pos, 3)
    r = eval_pos-dipole_pos_array # relative distance to dipole
    
    r_abs = np.linalg.norm(r, axis=1)
    r_abs_array = np.tile(r_abs, 3).reshape(3, n_pos).T
    unit_position = r/r_abs_array
    
    phase_factor_positional = np.exp(1j*ang_freq*r_abs_array/constants.c)
    phase_factor_temporal = np.exp(-1j*ang_freq*time)
    phase_factor = phase_factor_positional*phase_factor_temporal
    
    cross_term = np.cross(unit_position, polarization)
    
    far_field = (1/r_abs_array)*cross_term
    b_field = prefactor*(far_field)*phase_factor
    
    return b_field
