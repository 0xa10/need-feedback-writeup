import numpy as np

matpow = np.linalg.matrix_power
def bitarray_to_dec(ba):
    """Takes a bit string (little endian) and converts it to a number"""
    acc = 0
    for i in range(len(ba)):
        acc += ba[i] * (2**i)
    return acc

def get_lfsr_equations(coeff, count):
    """Returns the LFSR equations for an lfsr with the given coeffecient vector.
    Will return @{count} equations"""
    # Set up our parameters 
    #   n - lfsr size
    #   m - base matrix
    #   coeff_m - multiplier matrix (with coefficients)
    #   X   -
    n = len(coeff)
    m = np.hstack((np.zeros((n, 1)), np.eye(n)[:,:n-1]))
    coeff_m = np.vstack((np.zeros((n-1, n)), coeff))
    #X = np.matrix((m + m @ coeff_m) % 2)
    
    tmp = np.eye(n)
    
    C = []
    for i in range(count):
        C.append(tmp[:,n-2])
        tmp = (tmp@m + tmp@m@coeff_m) % 2
        # C = np.hstack((C, (matpow(X, i) % 2)[:,n-2]))

    return np.matrix(C).transpose()[:n-1,:]

def solve_lfsr(coeff, S):
    n = len(coeff)
    
    C = get_lfsr_equations(coeff, n-1)
    
    Stemp = np.hstack((np.matrix([0,]), np.matrix(S))) # Add dummy result for first bit
    Ctemp = np.hstack((np.matrix(np.eye(n))[:,n-1], C)) # Add dummy column for first bit

    A = np.linalg.solve(Ctemp.transpose(), Stemp.transpose()) % 2

    return bitarray_to_dec(np.array(A.transpose(),dtype=int)[0].tolist())    

