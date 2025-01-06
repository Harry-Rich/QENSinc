import numba as numba
import numpy as np


@numba.jit(fastmath=True, nopython=True, parallel=True)
def calculate_Finc_qt(q_points: np.ndarray, disp_3d: np.ndarray,
                      dt: np.ndarray,
                      total_scat_lengths: np.ndarray) -> np.ndarray:
    """
    Calculate the incoherent intermediete scattering function F(q,t) as per eqn17 in https://doi.org/10.1016/0010-4655(95)00048-K

    :param q_points: Chosen q_points in 3 dimensional cartesian coordinates

    :returns: Numpy array [q_points x time_interval] giving the incoherent intermediate scattering function Finc(q,t)
    
    """

    incoh_f = np.zeros((q_points.shape[0], dt.shape[0]), dtype=np.complex64)

    s_len_sq = np.zeros(total_scat_lengths.shape, dtype=np.float64)

    for x in numba.prange(0, s_len_sq.shape[0]):
        s_len_sq[x] = total_scat_lengths[x]**2

    for k in numba.prange(0, len(q_points)):
        for i in numba.prange(0, len(dt)):

            # numba dot prod implementation
            result = np.zeros(disp_3d[i].shape[0:2])
            for x in numba.prange(3):
                result += disp_3d[i][:, :, x] * q_points[k][x]

            #numba raise to exponent implementation
            expensive_exponent = np.zeros(result.shape, dtype=np.complex64)
            for x in numba.prange(result.shape[0]):
                for y in numba.prange(result.shape[1]):
                    expensive_exponent[x, y] = np.exp(1j * result[x, y])

            # Numba axis mean implementation
            mean_1 = np.zeros(expensive_exponent.shape[0], dtype=np.float64)
            for y in numba.prange(expensive_exponent.shape[1]):
                for x in numba.prange(expensive_exponent.shape[0]):
                    mean_1[x] += np.real(expensive_exponent[x, y])

            # Numba mean implementation
            for x in numba.prange(mean_1.shape[0]):
                mean_1[x] = mean_1[x] / expensive_exponent.shape[1]

            # Numba mean implementation
            incoh_val = 0
            for x in numba.prange(mean_1.shape[0]):
                incoh_val += mean_1[x] * s_len_sq[x]
            incoh_f[k, i] = incoh_val / mean_1.shape[0]

    return incoh_f
