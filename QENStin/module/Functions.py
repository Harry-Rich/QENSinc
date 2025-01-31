import numba as numba
import numpy as np



@numba.njit
def create_disp_3d(disp, i, unit='A'):
    """
    From a displacement array calculate the 3d displacements for a given time interval
    """
    n, m, t = disp.shape
    # Initialize the output array
    result = np.zeros((m, n-i, t), dtype=disp.dtype)
    # Compute the displacements
    result[:, 0, :] = disp[i, :, :]  # Base case
    for j in numba.prange(1, n - i):
        result[:, j, :] = disp[i+j,:, :] - disp[j-1,:, :]
    
    if unit = 'A':
        return result
    elif unit = 'nm':
        return result / 10
    else:
        raise ValueError('unit must be either A or Nm')


@numba.jit(fastmath=True, nopython=True, parallel=True)
def mat_mul(a,b):
    """
    3d x 2d matrix multiplication using numba

    i.e. 
    a is a 3d array of shape (n,m,t)
    b is a 2d array of shape (t,p)

    The result is a 3d array of shape (n,m,p) where the last dimension is the dot product of the last dimension of a and b
    """
    result_shape = a.shape[:-1] + (b.shape[0],) 
    result = np.zeros(result_shape,dtype = np.float64)

    for i in numba.prange(a.shape[0]):  
        for j in numba.prange(a.shape[1]):  
            for k in numba.prange(b.shape[0]): 
                for l in numba.prange(a.shape[2]):  # Inner dimension for dot product
                    result[i, j, k] += a[i, j, l] * b[k, l]
    return result


@numba.jit(fastmath=True, nopython=True, parallel=True)
def mean_axis_2(array):
    """
    Mean along the 2nd axis of a 3d array
    
    """

    result = np.zeros((array.shape[0],array.shape[1]), dtype = np.float64)
    array = array.real

    for x in numba.prange(array.shape[0]):
        for y in numba.prange(array.shape[1]):
            for z in numba.prange(array.shape[2]):
                result[x,y] += array[x,y,z]
            result[x,y] = result[x,y] / array.shape[2]

    return result


@numba.jit(fastmath=True, nopython=True, parallel=True)
def calc_incoh_f(disp_3d:np.ndarray,
         q_points: np.ndarray, 
         s_len_sq: np.ndarray) -> np.ndarray:
    """
    Calculate the incoherent intermediete scattering function F(q,t) as per eqn17 in https://doi.org/10.1016/0010-4655(95)00048-K

    :param q_points: Chosen q_points in 3 dimensional cartesian coordinates, shape [q_points x 3]

    :returns: Point along F(q_t) line for the chosen 3d displacement and q_point

    expects an input of X atoms by N observations by 3 dimensions


    """
    result = mat_mul(disp_3d,q_points)

    #numba raise to exponent implementation
    expensive_exponent = np.zeros(result.shape, dtype=np.complex64)
    for x in numba.prange(result.shape[0]):
        for y in numba.prange(result.shape[1]):
            expensive_exponent[x, y] = np.exp(1j * result[x, y])


    # Numba axis mean implementation
    mean_1 = mean_axis_2(expensive_exponent)

    # Numba sum over implementation and divide by N
    incoh_val = np.zeros(1,dtype = np.float64)
    for x in numba.prange(mean_1.shape[0]):
        for y in numba.prange(mean_1.shape[1]):
            incoh_val += mean_1[x,y] * s_len_sq[x]

    incoh_f = incoh_val / (mean_1.shape[0]*mean_1.shape[1])

    return incoh_f


def incoh_line(disp,q_points,dt,total_scat_lengths,min_required_points= 10000, unit = 'A'):
    """
    Calc the incoherent intermediate scattering function F(q,t) for a given 3d displacement and q_shell, based on min required points.
    Input of non square scattering lengths is expected

    """

    s_len_sq = np.zeros(total_scat_lengths.shape, dtype=np.float64)
    for x in range(0, s_len_sq.shape[0]):
        s_len_sq[x] = total_scat_lengths[x]**2

    incoh_line = np.zeros((dt.shape[0] - min_required_points), dtype=np.float64)

    for i in range(0, (dt.shape[0]-min_required_points)):
        if i % 3000 == 0:
            print(i)
        disp_3d = create_disp_3d(disp,i,unit)
        incoh_line[i] = calc_incoh_f(disp_3d,q_points,s_len_sq)

    return incoh_line
