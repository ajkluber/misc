import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def count_transitions(int n_bins, int n_steps, np.ndarray[np.double_t, ndim=1] bins, np.ndarray[np.double_t, ndim=1] Q):
    """ Count the number of transitions between bins """
    cdef np.ndarray[np.double_t,
                    ndim=2,
                    negative_indices=False,
                    mode='c'] N = np.zeros((n_bins,n_bins))

    cdef int i,j,t

    for t in range(n_steps):
        for i in range(n_bins+1):
            if (bins[i] < Q[t]) and (bins[i+1] >= Q[t]):
                if t == 0:
                    j = i
                else:
                    N[i,j] += 1.
                    j = i
                break
    return N

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_detailed_balance_R(int n_bins, np.ndarray[np.double_t, ndim=2] Rij, np.ndarray[np.double_t, ndim=1] P):
    """Calculates the diagonal and superdiagonal of rate matrix using detailed balance"""
    cdef int j
    for j in range(n_bins - 1):
        Rij[j,j + 1] = Rij[j + 1,j]*P[j + 1]/P[j]

    for j in range(n_bins):
        if j == 0:
            Rij[j,j] = -Rij[j + 1,j]
        elif j == (n_bins - 1):
            Rij[j,j] = -Rij[j - 1,j]
        else:
            Rij[j,j] = -(Rij[j - 1,j] + Rij[j + 1,j])
    return Rij

#@cython.boundscheck(False)
#@cython.wraparound(False)
#def calculate_detailed_balance_R(int n_bins, np.ndarray[np.double_t, ndim=2] Rij, np.ndarray[np.double_t, ndim=1] P):
#    """Constrain matrix to fulfill detailed balance. Allow all elements"""
#    cdef int i,j
#    cdef np.double_t row_sum = 0.
#
#    for j in range(n_bins - 1):
#        for i in range(j + 1,n_bins):
#            Rij[i,j] = Rij[j,i]*P[j]/P[i]
#        row_sum = 0.
#        for i in range(n_bins):
#            if (i != j):
#                row_sum += Rij[j,i]
#        Rij[j,j] = -row_sum
#    Rij[n_bins - 1,n_bins - 1] = -np.sum(Rij[n_bins - 1,:n_bins - 1])
#
#    return Rij

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_logL_smoothed(int n_bins, np.ndarray[np.double_t, ndim=2] Nij, \
                        np.ndarray[np.double_t, ndim=2] exp_tRij, \
                        np.ndarray[np.double_t, ndim=2] Rij, \
                        np.ndarray[np.double_t, ndim=1] P, \
                        np.double_t deltaQ, np.double_t gamma):

    cdef np.ndarray[np.double_t,
                    ndim=1,
                    negative_indices=False,
                    mode='c'] D = np.zeros(n_bins-1)

    cdef int i,j
    cdef np.double_t neg_lnL = 0.
    cdef np.double_t D2_sum = 0.

    # Calculate -lnL. Sum over all bins in the transition matrix
    for i in range(n_bins):
        for j in range(n_bins):
            neg_lnL += Nij[i,j]*np.log(exp_tRij[i,j])
    neg_lnL = -1*neg_lnL

    # Add the smoothening part to -lnL
    for i in range(n_bins - 1):
        D[i] = (deltaQ**2)*Rij[i + 1,i]*np.sqrt(P[i]/P[i + 1])

    for i in range(n_bins - 2):
        D2_sum += (D[i] - D[i + 1])**2

    neg_lnL = neg_lnL + (0.5/(gamma**2))*D2_sum

    return neg_lnL

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_logL_smoothed_FD(int n_bins, np.ndarray[np.double_t, ndim=2] Nij, \
                        np.ndarray[np.double_t, ndim=2] Propagator, \
                        np.ndarray[np.double_t, ndim=1] D, np.double_t gamma):

    cdef int i,j
    cdef np.double_t neg_lnL = 0.
    cdef np.double_t D2_sum = 0.

    # Calculate -lnL. Sum over all bins in the transition matrix
    for i in range(n_bins):
        for j in range(n_bins):
            neg_lnL += Nij[i,j]*np.log(Propagator[i,j])
    neg_lnL = -1*neg_lnL

    # Add the smoothening part to -lnL
    for i in range(n_bins - 2):
        D2_sum += (D[i] - D[i + 1])**2

    neg_lnL = neg_lnL + (0.5/(gamma**2))*D2_sum

    return neg_lnL

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_logL(int n_bins, np.ndarray[np.double_t, ndim=2] Nij, np.ndarray[np.double_t, ndim=2] exp_tRij):

    cdef int i,j
    cdef np.double_t neg_lnL = 0.

    # Calculate -lnL. Sum over all bins in the transition matrix
    for i in range(n_bins):
        for j in range(n_bins):
            neg_lnL += Nij[i,j]*np.log(exp_tRij[i,j])
    neg_lnL = -1*neg_lnL

    return neg_lnL

#@cython.boundscheck(False)
#@cython.wraparound(False)
#def calculate_M(int n_bins, np.ndarray[np.double_t, ndim=1] F, np.ndarray[np.double_t, ndim=1] D, np.double_t deltaQ):
#    """ Calculate matrix M from Bicout, Szabo 1998. 109"""
#
#    cdef np.ndarray[np.double_t,
#                    ndim=2,
#                    negative_indices=False,
#                    mode='c'] M = np.zeros((n_bins,n_bins))
#    cdef int i
#
#    for i in range(n_bins):
#        if i == 0:
#            M[i,i] = -((D[i+1] + D[i])/(2.*deltaQ**2))*np.exp(-(F[i] - F[i+1])/2.)
#            M[i,i+1] = np.sqrt(((D[i+1] + D[i])/(2.*deltaQ**2))*np.exp(-(F[i] - F[i+1])/2.)*((D[i] + D[i+1])/(2.*deltaQ**2))*np.exp(-(F[i+1] - F[i])/2.))
#            M[i+1,i] = M[i,i+1]
#        elif i == (n_bins-1):
#            M[i,i] = -((D[i-1] + D[i])/(2.*deltaQ**2))*np.exp(-(F[i] - F[i-1])/2.)
#            M[i,i-1] = np.sqrt(((D[i-1] + D[i])/(2.*deltaQ**2))*np.exp(-(F[i] - F[i-1])/2.)*((D[i] + D[i-1])/(2.*deltaQ**2))*np.exp(-(F[i-1] - F[i])/2.))
#            M[i-1,i] = M[i,i-1]
#        else:
#            M[i,i] =-((D[i+1] + D[i])/(2.*deltaQ**2))*np.exp(-(F[i] - F[i+1])/2.) - ((D[i-1] + D[i])/(2.*deltaQ**2))*np.exp(-(F[i] - F[i-1])/2.)
#            M[i,i+1] = np.sqrt(((D[i+1] + D[i])/(2.*deltaQ**2))*np.exp(-(F[i] - F[i+1])/2.)*((D[i] + D[i+1])/(2.*deltaQ**2))*np.exp(-(F[i+1] - F[i])/2.))
#            M[i,i-1] = np.sqrt(((D[i-1] + D[i])/(2.*deltaQ**2))*np.exp(-(F[i] - F[i-1])/2.)*((D[i] + D[i-1])/(2.*deltaQ**2))*np.exp(-(F[i-1] - F[i])/2.))
#            M[i+1,i] = M[i,i+1]
#            M[i-1,i] = M[i,i-1]
#
#    return M
#
#@cython.boundscheck(False)
#@cython.wraparound(False)
#def calculate_propagator(int n_bins, np.ndarray[np.double_t, ndim=2] M, np.ndarray[np.double_t, ndim=1] F, np.double_t dt):
#
#    cdef np.ndarray[np.double_t,
#                    ndim=2,
#                    negative_indices=False,
#                    mode='c'] P = np.zeros((n_bins,n_bins))
#
#    cdef np.ndarray[np.double_t,
#                    ndim=2,
#                    negative_indices=False,
#                    mode='c'] vects = np.zeros((n_bins,n_bins))
#
#    cdef np.ndarray[np.double_t,
#                    ndim=1,
#                    negative_indices=False,
#                    mode='c'] vals = np.zeros(n_bins)
#
#    cdef int i,j
#
#    vals, vects = np.linalg.eig(M)
#
#    for i in range(n_bins):
#        vects[:,i] = vects[:,i]/np.linalg.norm(vects[:,i])
#
#    for i in range(n_bins):
#        for j in range(n_bins):
#            for alpha in range(n_bins):
#                P[i,j] += np.exp(-(F[j] - F[i])/2.)*vects[i,alpha]*vects[j,alpha]*np.exp(vals[alpha]*dt)
#
#    return P
