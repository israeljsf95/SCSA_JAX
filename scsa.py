import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from scipy.sparse import spdiags
from scipy.special import gamma
from scipy.integrate import simps
from functools import partial

from jax.config import config
config.update("jax_enable_x64", True)

# Configuration for pseudo-number-generation
from jax import random
key = random.PRNGKey(758493)  # Random seed is explicit in JAX
#random.uniform(key, shape=(1000,))


def gen_sig(M:int, L:int, sig_type:int, noise:bool = False):
    """
    This function is a signal generator.
        M: Is integer represeting the number of points used to discretize
        L: is the interval Length (by default, the SCSA assumes L as multiple of 2*pi)
        sig_type: is the type of signal required for default benchmarking and test cases with SCSA
            1: sinusoid
            2: peak - signal
            3: 4 peaks - signal
        noise: is a bool variable that enables adding noise to the output of each signal

    Return:
        y: clean signal if noise is False
        y_noise: corrupted signal if noise True
    """
    def sech(x):
        """
        Function that computes the hyperbolic secant of given discrete grid
        """
        return 1 / jnp.cosh(x)

    match sig_type:

        case 1:

            M = 2**12
            a, b = -jnp.pi, jnp.pi
            dt = (b - a) / (M-1)
            t = jnp.arange(a, b, dt)
            f0 = 1
            y = jnp.sin(2*jnp.pi*f0*t)
            if noise:
                n = 0.1*random.normal(key, shape = (len(y), ))
                y = y + n
            return t, y
        case 2:
            h = jnp.pi*2 / M
            x = h*np.arange(1, M+1)
            x = L*(x - jnp.pi)/jnp.pi
            y = 60*sech(x - 2)**2
            if noise:
                n = 5*random.normal(key, shape = (len(y), ))
                y = y + n
            return x, y

        case 3:
            h = jnp.pi*2 / M
            x = h*np.arange(1, M+1)
            x = L*(x - jnp.pi)/jnp.pi
            y = 120*sech(x + 8)**2 + 120*sech(x + 2)**2 + 65*sech(x - 2)**2 + 65*sech(x - 8)**2
            if noise:
                n = 8*random.normal(key, shape = (len(y), ))
                y = y + n

            return x, y
        case  _:
            M = 2**12
            a, b = -jnp.pi, jnp.pi
            dt = (b - a) / (M-1)
            t = jnp.arange(a, b, dt)
            f0 = 40
            y = jnp.sin(2*jnp.pi*f0*t)
            if noise:
                n = random.normal(key, shape = (len(y), ))
                y = y + n
            return t, y


def construct_matrix(n:int, fs:float = 1):

    """
    Function that construct the Second Derivative Operator following the previous implementations of SCSA
        n: integer that represent the signal length, i.e, the length of the discretized space
        fs: the sample frequency (the previous SCSA implementation assumes fs = 1)

    Return
        D: a matrix (n x n) that represents the second order derivative operator used in the hamiltonian construction for
           SCSA
    """

    fsh = 2*jnp.pi/n
    aux_1 = jnp.arange(n-1, 0, -1)
    aux_2 = jnp.ones((n, 1))
    aux_cte_1 = jnp.pi**2/(3*fsh**2)
    D = jnp.kron(aux_1, aux_2)


    if n % 2 == 0:
        sines = jnp.sin(D*fsh*0.5)**2
        msines = jnp.sin(-D*fsh*0.5)**2
        dx = -aux_cte_1 - (1/6)*aux_2
        test_bx = -(-1)**(D) * .5 / sines
        test_tx = -(-1)**(-D) * .5 / msines
    else:
        sines = jnp.sin(D*fsh*0.5)**2
        msines = jnp.sin(-D*fsh*0.5)**2
        dx = -aux_cte_1 - (1/12) * aux_2
        test_bx = -0.5*((-1)**D) * jnp.cot(D*fsh*0.5) / sines
        test_tx = -0.5*((-1)**(-D)) * jnp.cot((-D)*fsh*0.5) / msines

    aux_matrix = jnp.block([test_bx, dx, test_tx])

    aux_vector = jnp.concatenate((jnp.arange(-(n-1), 1, 1), jnp.arange(n-1, 0,-1)), axis = 0)
    aux_matrix_final = spdiags(aux_matrix.T, aux_vector, n, n)
    aux_matrix_final = jnp.array(aux_matrix_final.toarray())

    Dx = (fsh/fs)**2 * aux_matrix_final
    return Dx

def simp_integral(y:jnp.array, dt:float):

    """
    Function that computes the numerical integral from simpsons method
        y: is the function array (each y_i represents a discrete sample from the function in the discrete time int(K*dt))
           (assumes as a matrix because: the squared eigenfunctions from Schrodinger Operator.)
        dt: is the time-sample
    Return
        I: integral value
    """
    n, _ = y.shape

    if n > 1:

        I = (1 / 3) * ( y[0, :] + y[1, :] )*dt
        for i in range(2, n):
            if i % 2 == 0:
                I += (1/3)*(y[i-1, :] + y[i, :])*dt
            else:
                I += (y[i-1, :] + (1/3)*y[i, :])*dt
    else:
        I = y*dt
    return I

def compute_scsa_1d(h:float , y:jnp.array, fs:float):
    """
    Function that computes the 1-d SCSA algorithm
        h: is the semi-classical parameter
        y: is the discrete signal
        fs: is sample frequency

    Return:
        yscsa: the reconstructed signal from the schroedinger operator
        neg_sch_evals: kappas (negative eigen_values)
        Nh: number of negative eigenvalues
        psinnorm: the normalized-squared eigenfunctions
    """
    n = len(y)
    y = y - y.min()
    gm = 0.5
    Lcl = (1/(2*(jnp.pi)**.5))*(gamma(gm+1)/gamma(gm + 1.5))
    D = construct_matrix(n, fs)
    D = -(h**2)*D
    schro_ope = D - jnp.diag(y)
    [sch_evals, sch_evec] = jla.eigh(schro_ope)
    Nh = jnp.sum(sch_evals < 0)
    neg_sch_evals = jnp.diag((-sch_evals[:Nh]) ** gm)
    neg_sch_evec = sch_evec[:, :Nh]
    aux_norm_psi = simp_integral(neg_sch_evec**2, fs)
    # aux_norm_psi = simps(neg_sch_evec ** 2, dx = fsh, axis = 0)
    # be careful here -- broadcast the integral of the squared eigenfunctions
    psinnorm = neg_sch_evec / (aux_norm_psi.reshape(1, -1) ** 0.5)
    yscsa = ((h/Lcl)* jnp.sum((psinnorm**2) @ neg_sch_evals, 1)) ** (2 / (1 + 2*gm))

    if y.shape != yscsa.shape:
        return yscsa.T, neg_sch_evals, Nh, psinnorm
    else:
        return yscsa, neg_sch_evals, Nh, psinnorm

def compute_cscsa_cost(y:jnp.array,  yscsa:jnp.array, mu = None):

    """
    Function that computes the C-SCSA cost function
        y: original potential, aka, discrete signal used as input to the algorithm
        yscsa: the reconstructino provided by SCSA

    Return:
        a scalar value that is the cost function for the SCSA method
    """

    first_term = jnp.mean((y - yscsa)**2)
    if mu is None:
        mu = 100*(y.max() / y.sum())
    else:
        mu = mu
    ## Curvature penalty
    yscsa_first_diff = jnp.gradient(yscsa)
    yscsa_second_diff = jnp.gradient(yscsa_first_diff)
    curvature_C = jnp.abs(yscsa_second_diff) / (1 + yscsa_first_diff**2)**1.5
    c_curv = jnp.sum(curvature_C)
    second_term = mu * c_curv # Penalizing term

    return first_term + second_term

def compute_scsa_denoising(y: jnp.array, fs:float,  verbose:bool = True, hsearch_method:str = "else", mu = None):

    """
    Function that performs an optimization problem by searching for the best h(Semi Classical parameter)
    that provides the best reconstruction of the potential, aka the discrete signal, in the sense of denoising
    and smooth characteristics of the potential
        y: the potential/signal to be denoised
        fs: the sample frequency
        verbose: parameter for displaying the iteration counter and the cost at i-th iteration
        hsearch_method: string that will represent the strategy for generating the search space

    Return:
        ysca: the reconstructed potential
        kappas: the eigenvalues of the SCSA operator
        Nh: Number of Negative eigenvalues
        psinNorm: Squared eigenfunctions
        cost_function: list with the cost function values computed at i-th step of the optimization process
    """

    n = len(y)
    hh_list = compute_hsearch(y, dt, method = hsearch_method)
    cost_function = []
    cont = 0
    #compute_scsa_1d_jit = jax.jit(compute_scsa_1d, static_argnums=(0, 1, 2, 3, 4))
    for i in tqdm(range(len(hh_list))):
        hh = hh_list[i]

        # yscsa, neg_sch_evals, Nh, psinnorm = compute_scsa_1d_jit(hh, n, y, fs, fsh)
        yscsa, neg_sch_evals, Nh, psinnorm = compute_scsa_1d(hh, y, fs)
        cost_function.append(compute_cscsa_cost(y, yscsa, mu))
        if verbose:
            print(f"Cost Function (C-SCSA) iter {cont + 1}: h == {hh:.3f} || J[{cont+1}] == {cost_function[-1]:.5f}")

    hh_min = hh_list[np.argmin(cost_function)]
    yscsa, kappas, Nh, psinNorm = compute_scsa_1d(hh_min, y, fs)
    return yscsa, kappas, Nh, psinNorm, cost_function


def compute_fourier_ssb(y:jnp.array, ts:float):

    """
    Function that computes the single side band (ssb) of the fourier transform for the signal
        y: the potential which we require the fourier coefficients
        ts: sample time used to generate the signal

    Return:
         |cn|: the absolute values of first half of the fourier spectrum
    """

    L = 2*jnp.pi
    M = len(y)
    xx, yy = jnp.meshgrid(jnp.arange(0, M), jnp.arange(0, M))
    w = jnp.exp(-1j*2*jnp.pi/M)
    dft_matrix = w**(xx * yy)
    y = y.reshape(-1, )
    cn = (ts / (2*L))*(dft_matrix @ y)

    #single sided band fourier_transform of the signal
    return jnp.abs(cn[0: (M // 2) + 1])

def h_bounds_evangelos(y: jnp.array, ts:float):
    """
    Function that computes the bounds for search the semi-classical parameter based on the master thesis of
    Evangelos Piliouras (alumni EMANG)
        y: the discretized potential
        ts: the sample time
    Return:
         h_min: minima value for search the semi-classical parameter
         h_max: maximum value for search the semi-classical parameter
    """

    M = len(y)
    cn = compute_fourier_ssb(y, ts)
    aux_cn = np.arange(1, len(cn))
    aux = jnp.sum(cn[1:]**2 / (2*(aux_cn**2)))

    if y.max() > cn[0]:
        h_max = (ts/jnp.pi)*M*jnp.sqrt(aux / (y.max() - cn[0]))
    else:
        h_max = (ts/jnp.pi)*M*jnp.sqrt(aux / (cn[0] - y.max()))
    h_min = (ts/jnp.pi)*jnp.sqrt(y.max())
    return h_min, h_max


def compute_hsearch(y, dt = 1, method = "else"):
    """
    Function that generates the search space for the semi-classical parameter.
    y: discrete potential
    dt: sample time
    method: string variable used to choose the heuristics
        "uniform": the search will be done in a uniform fashion h belongs to the interval [a, b]
        "evangelos_": the search will be done in a smarter fashion where h belongs to the interval [h_min, h_max]
        "else": the search will be done in a heuristic way provided by the previous implementations of the method
    return:
        hh: list that contains the values of the semi-classical parameter used in the search.
    """
    y = jnp.real(y)
    y_max = y.max()
    if method == "uniform":
        hh = jnp.arange(1, 30, dt)
    elif method == "evangelos_" :
        #Evangelos Search
        h_min, h_max = h_bounds_evangelos(y, dt)
        hh = jnp.arange(h_min, h_max, dt)
    else:
        h_min = (dt/jnp.pi)*jnp.sqrt(y.max())
        hh = jnp.arange(h_min, y_max, dt)

    return hh

if __name__ == "__main__":

    # Testing for the generation of the Matrix Operator
    # n = 10
    # fs = 1
    # fsh = 2*jnp.pi / n
    # D = construct_matrix(n, fs, fsh)
    # print(D)
    M, L = 512, 8*jnp.pi
    sig_type = 3
    x, y = gen_sig(M, L, sig_type)
    x, y_noise = gen_sig(M, L, sig_type, noise = True)
    fs = 1
    fig, axs = plt.subplots(2, 1)
    axs = axs.flatten()
    axs[0].plot(x, y)
    axs[0].set_title("Clean Potential")
    axs[1].plot(x, y_noise)
    axs[1].set_title("Noised Potential")
    plt.tight_layout()
    plt.show()


    dt = 1
    #two different test cases
    mu = 0
    #mu = None
    print("Computing SCSA for the clean potential: ")
    yscsa_clean, kappas, Nh, psinNorm, cf_clean = compute_scsa_denoising(y, fs,
                                                                         verbose = False, mu = mu)

    print()
    print("Computing SCSA for the noised potential: ")
    yscsa_denoised, kappas_nn, Nh_nn, psinNorm_nn, cf_noise = compute_scsa_denoising(y_noise, fs,
                                                                                     verbose = False, mu = mu)



    fig, axs = plt.subplots(1)
    axs.plot(cf_clean, label = "Clean Cost Function")
    axs.set_title("Cost Functions")
    axs.plot(cf_noise, label = "Noised Cost Function")
    axs.legend()
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 1)
    axs = axs.flatten()
    axs[0].plot(x, y, label = "Clean Signal")
    axs[0].plot(x, yscsa_clean, label = "Reconstructed Signal")
    axs[0].set_title("Clean Potential")
    axs[1].plot(x, y_noise, label = "Noised Signal")
    axs[1].plot(x, yscsa_denoised, label = "Reconstructed Noised Signal")
    axs[1].set_title("Noised Potential")
    plt.tight_layout()
    plt.show()

#%%
