# --------------------------------
# Helper function for PC evaluation 
# --------------------------------
from ufl import *
from dolfin import *
import numpy as np 

import os
os.environ['MPLCONFIGDIR'] = "."


####################################################
# Method for analytical calculation of eigenvalues #
####################################################
def eigenValues(T, tol=DOLFIN_EPS):
    '''
    Method to calculate the (real) eigenvalues of a second order tensor T using
    its principal invariants.
    '''
    dim=T.geometric_dimension()                                                 # get geometric dimension of tensor T to decide which subroutine to use for eigenvalue computations

    # choose projection tensor routine based on dimensionality of problem
    if dim == 2:                                                                # 2D-problem
        return _eigVal2D(T,tol)
    else:                                                                       # 3D-problem
        return _eigVal3D(T,tol)


########################################################################
# Method for the analytical calculation of eigenvalues for 2D-Problems #
########################################################################
def _eigVal2D(T, tol):
    '''
    Analytically calculate eigenvalues for a two-dimensional tensor with a
    characteristic polynomial equation of the form

                    lambda^2 - I1*lambda + I3 = 0   .

    Since the characteristic polynomial is a reduced quadratic equation, the
    eigenvalues can be determined using the p-q-formula.

    NOTE:
    The method implemented here, implicitly assumes that the polynomial has
    only real roots, since imaginary ones should not occur in this use case.

    In order to ensure eigenvalues with algebraic multiplicity of 1, the idea
    of numerical perturbations is adopted from "Computation of isotropic tensor
    functions" by C. Miehe (1993). Since direct comparisons with conditionals
    have proven to be very slow, not the eigenvalues but the coefficients
    occuring during the calculation of them are perturbated to get distinct
    values.
    '''

    # determine perturbation from tolerance
    pert = 2*tol

    # get required invariants
    I1 = tr(T)                                                               # trace of tensor
    I3 = det(T)                                                              # determinant of tensor

    # determine discriminant
    # -> If the discriminant is 0, eigenvalues with a algebraic multiplicity
    # -> of 2 could occur. Here, this is prevented by "artificially creating"
    # -> eigenvalues with an algebraic multiplicity of 1 by adding a numerical
    # -> perturbation to discriminants that are close to 0. This is especially
    # -> necessary since a zero discriminant will cause problems during the
    # -> automatic differentiation procedure
    D = I1**2-4*I3                                                              # preliminary value for discriminant
    D=conditional(lt(D,tol),abs(D)+pert,D)                              # add numerical perturbation to discriminant, if close to zero; ensure positiveness of D

    # calculate polynomial roots
    # -> Here, the polynomial solution formula is only used to calculate the
    # -> "bigger" root, since - if I1 and sqrt(D) are of similar magnitude - a
    # -> loss of significance could lead to erroneous results. For the second
    # -> root, Vietas formula is used.
    lambda1 = 0.5*(I1+sqrt(D))
    lambda2 = I3/lambda1

    # return polynomial roots (eigenvalues)
    return lambda1, lambda2


########################################################################
# Method for the analytical calculation of eigenvalues for 3D-Problems #
########################################################################
def _eigVal3D(T, tol):
    '''
    Analytically calculate eigenvalues for a three-dimensional tensor T with a
    characteristic polynomial equation of the form

                lambda^3 - I1*lambda^2 + I2*lambda - I3 = 0   .

    Since the characteristic polynomial is in its normal form , the eigenvalues
    can be determined using Cardanos formula. This algorithm is based on:
    "Efficient numerical diagonalization of hermitian 3 × 3 matrices" by
    J. Kopp (equations: 21-34, with coefficients: c2=-I1, c1=I2, c0=-I3).

    NOTE:
    The method implemented here, implicitly assumes that the polynomial has
    only real roots, since imaginary ones should not occur in this use case.

    In order to ensure eigenvalues with algebraic multiplicity of 1, the idea
    of numerical perturbations is adopted from "Computation of isotropic tensor
    functions" by C. Miehe (1993). Since direct comparisons with conditionals
    have proven to be very slow, not the eigenvalues but the coefficients
    occuring during the calculation of them are perturbated to get distinct
    values.
    '''

    # determine perturbation from tolerance
    pert = 2*tol

    # get required invariants
    I1 = tr(T)                                                               # trace of tensor
    I2 = 0.5*(tr(T)**2-inner(T,T))                                        # 2nd invariant of tensor
    I3 = det(T)                                                              # determinant of tensor

    # determine terms p and q according to the paper
    # -> Follow the argumentation within the paper, to see why p must be
    # -> positive. Additionally ensure non-zero denominators to avoid problems
    # -> during the automatic differentiation
    p = I1**2 - 3*I2                                                            # preliminary value for p
    p = conditional(lt(p,tol),abs(p)+pert,p)                            # add numerical perturbation to p, if close to zero; ensure positiveness of p
    q = 27/2*I3 + I1**3 - 9/2*I1*I2                                             # preliminary value for q
    q = conditional(lt(abs(q),tol),q+sign(q)*pert,q)                # add numerical perturbation (with sign) to value of q, if close to zero

    # determine angle phi for calculation of roots
    phiNom2 =  27*( 1/4*I2**2*(p-I2) + I3*(27/4*I3-q) )                         # preliminary value for squared nominator of expression for angle phi
    phiNom2 = conditional(lt(phiNom2,tol),abs(phiNom2)+pert,phiNom2)    # add numerical perturbation to ensure non-zero nominator expression for angle phi
    phi = 1/3*atan_2(sqrt(phiNom2),q)                                   # calculate angle phi

    # calculate polynomial roots
    lambda1 = 1/3*(sqrt(p)*2*cos(phi)+I1)
    lambda2 = 1/3*(-sqrt(p)*(cos(phi)+sqrt(3)*sin(phi))+I1)
    lambda3 = 1/3*(-sqrt(p)*(cos(phi)-sqrt(3)*sin(phi))+I1)

    # return polynomial roots (eigenvalues)
    return lambda1, lambda2, lambda3


##############################################################
# Method for an analytical calculation of projection tensors #
##############################################################
def projectionTensors(T, *eigenValues):
    '''
    Method to calculate the projection tensors of a second order tensor T using
    its already known eigenvalues.
    '''
    dim=T.geometric_dimension()                                                 # get geometric dimension of tensor T to decide which subroutine to use for projection tensor computations

    # choose projection tensor routine based on dimensionality of problem
    if dim == 2:                                                                # 2D-problem
        return _projTen2D(T,*eigenValues)
    else:                                                                       # 3D-problem
        return _projTen3D(T,*eigenValues)


##############################################################################
# Method for an analytical calculation of projection tensors for 2D-problems #
##############################################################################
def _projTen2D(T, lambda1, lambda2):
    '''
    Analytically calculate projection tensors for a two-dimensional tensor T
    with two eigenvalues of algebraic multiplicity 1 (distinct eigenvalues).

    Mathematical manipulations show that the projection tensors of a simplified
    two-dimensional problem can be computed independent of the third eigenvalue
    by just using the simplified version of the three-dimensional problem:

        M_a = T-lambda_b*I/(lambda_a-lambda_b)

    with a,b={1,2} and b!=a
    '''

    # get required quantities:
    I = Identity(2)

    # calculate individual projection tensors
    M1 = (T-lambda2*I)/(lambda1-lambda2)
    M2 = (T-lambda1*I)/(lambda2-lambda1)

    # return projection tensors
    return M1, M2


##############################################################################
# Method for an analytical calculation of projection tensors for 3D-problems #
##############################################################################
def _projTen3D(T, lambda1, lambda2, lambda3):
    '''
    Analytically calculate projection tensors for a three-dimensional tensor T
    with 3 eigenvalues of algebraic multiplicity 1 (distinct eigenvalues).

    Using some mathematical manipulations on the initial eigenvalue problem and
    the properties of projection tensors, they can be expressed in terms of the
    tensor T and its eigenvalues as "Frobenius-covariants"

        M_a = prod_b (T-lambda_b*I)/(lambda_a-lambda_b)

    for a,b={1,2,3} and b!=a.
    '''
    # get required quantities
    I = Identity(3)

    # calculate the individual projection tensors
    M1 = (T-lambda2*I)*(T-lambda3*I)/((lambda1-lambda2)*(lambda1-lambda3))
    M2 = (T-lambda3*I)*(T-lambda1*I)/((lambda2-lambda3)*(lambda2-lambda1))
    M3 = (T-lambda1*I)*(T-lambda2*I)/((lambda3-lambda1)*(lambda3-lambda2))

    # return projection tensors
    return M1, M2, M3
