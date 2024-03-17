import itertools
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from math import *
from scipy.stats import pearsonr


# generate g
def generate_g(N, R, mu=1, sigma=0.2):
    g = np.random.normal(mu, sigma, (N, R))
    cutoff = 0.99
    g = np.clip(g, a_min=mu-cutoff, a_max=mu+cutoff)
    return abs(g)

def generate_uni_g(N, R, lo=0.6, hi=1.4):
    g = np.random.uniform(lo, hi, (N, R))
    return g

# randomly generate pref orders (not used for smart bugs)
def make_preference_list(N, R):
    permutations = list(itertools.permutations(list(range(1, R+1))))
    preference_list = random.choices(permutations, k=N)
    return np.array(preference_list)

# generate preference orders for smart bugs with given g matrix
def smart_preference_list(g):
    return np.argsort(-g, axis=1) + 1

# a hand-waving pref order complementarity
def comp(pref, N):
    unique_counts = [len(np.unique(pref[:, col])) for col in range(pref.shape[1])]
    return sum(unique_counts)/N**2

# separate complementarity on each resource
def comp_sep(pref, N):
    unique_counts = [len(np.unique(pref[:, col]))/N for col in range(pref.shape[1])]
    return unique_counts

# check allowed depletion orders for a certain set of bugs
def allowed_orders(pref_list):
    R = len(pref_list[0])
    permutations = list(itertools.permutations(list(range(1, R+1))))
    allowed_orders_list = []
    for i in permutations:
        mark=0
        temp_list = [j for j in pref_list]
        while(mark<R):
            res_pool = [j[0] for j in temp_list]
            if(i[mark] not in res_pool):
                mark=-1
                break
            else:
                temp_list = [[k for k in j if k!= i[mark]] for j in temp_list]
                mark+=1
        if(mark != -1):
            allowed_orders_list.append(i)
    return allowed_orders_list

# for a given depletion order, generate G matrix
# here we only consider 1 season setup -- where nutrients just deplete one by one. 
def G_mat(g, pref, dep_order, N, R):
    G = np.zeros([N, R])
    for i_n in range(N):
        for i_t in range(R):
            for i in range(R):
                top_resource = pref[i_n][i]
                if(top_resource in dep_order[i_t:]):
                    break
            G[i_n, i_t] = g[i_n, top_resource-1]
    return G
# G_mat for co-utilizers
def G_mat_co(g, x, dep_order, N, R):
    G = np.zeros([N, R])
    for i_n in range(N):
        for i_t in range(R):
            present_res = dep_order[i_t:]-1
            if(sum(x[i_n, present_res])) == 0:
                G[i_n, i_t] = 0
            else: G[i_n, i_t] = g[i_n, present_res]@x[i_n, present_res] / sum(x[i_n, present_res])
    return G


# find the corresponding t values.
def t_niches(g, pref, dep_order, logD, N, R):
    G = G_mat(g, pref, dep_order, N, R)
    return np.linalg.inv(G)@np.ones(R)*logD

# make the resource-to-species conversion matrix based on t's. 
def F_mat(g, pref, dep_order, logD, N, R):
    F_mat = np.zeros([R, N])
    G = G_mat(g, pref, dep_order, N, R)
    t = np.linalg.inv(G)@np.ones(R)*logD
    for i_n in range(N):
        coeff = 1
        start = 0
        for i in range(R):
            if(start < R and pref[i_n][i] in dep_order[start:]):
                end = dep_order.index(pref[i_n][i]) + 1
                delta_t = sum(t[start:end])
                g_temp = g[i_n, pref[i_n][i]-1]
                F_mat[pref[i_n][i]-1, i_n] = coeff * (exp(g_temp*delta_t) - 1)
                coeff *= exp(g_temp*delta_t)
                start = end
            else:
                continue
    return F_mat

# F_mat for co utilizers
def F_mat_co(g, x, dep_order, logD, N, R):
    F_mat = np.zeros([R, N])
    G = G_mat_co(g, x, dep_order, N, R)
    t = np.linalg.inv(G)@np.ones(R)*logD
    for i_n in range(N):
        coeff = 1
        for i_t in range(R):
            present_res = dep_order[i_t:]-1
            for r in present_res:
                # here need to consider if g@x==0
                if( g[i_n, present_res]@x[i_n, present_res] == 0 ):
                    F_mat[r][i_n] += 0
                else:
                    F_mat[r][i_n] += coeff * (( x[i_n, r]*g[i_n, r] / (g[i_n, present_res]@x[i_n, present_res]) ) * (exp(G[i_n, i_t]*t[i_t]) - 1) )
            coeff *= exp(G[i_n, i_t]*t[i_t])
    return F_mat

# alternative F if we know t niches -- this is for R>N
def F_mat_alt(g, pref, dep_order, t, N, R):
    # print(g, pref, dep_order, t)
    F_mat = np.zeros([R, N])
    for i_n in range(N):
        coeff = 1
        start = 0
        for i in range(R):
            if(start < R and pref[i_n][i] in dep_order[start:]):
                end = dep_order.index(pref[i_n][i]) + 1
                delta_t = sum(t[start:end])
                g_temp = g[i_n, pref[i_n][i]-1]
                F_mat[pref[i_n][i]-1, i_n] = coeff * (exp(g_temp*delta_t) - 1)
                coeff *= exp(g_temp*delta_t)
                start = end
            else:
                continue
    return F_mat

# binning for plots. 
# input: one or many sets of (x, y). like:
# x = [[first set of var_x], [another set of var_x], ...]
# y = [[first set of var_y], ...]
# output: for each section in x, what is the mean and err of y. 
def binning(x, y, n_bins):
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
    bin_indices = np.digitize(x, bins)
    bin_means = np.array([[np.mean(y[j][bin_indices[j] == i]) for i in range(1, n_bins + 1)] for j in range(y.shape[0])])
    bin_err = np.array([[np.std(y[j][bin_indices[j] == i])/sqrt(len(y[j][bin_indices[j] == i])) for i in range(1, n_bins + 1)] for j in range(y.shape[0])])
    return bins, bin_means, bin_err

# binning but only for histogram.
# input: one or many sets of x.
# output: for each section of x, what is the frequency. 
def bin_hist(x, n_bins):
    x = np.atleast_2d(x)
    bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
    histlist = []
    for i in range(x.shape[0]):
        hist, _ = np.histogram(x[i, :], bins=bins)
        histlist.append(hist)
    return bins[:-1], np.array(histlist)


# dynamical stability mod for the diauxers
def b_to_b(g, dep_order, G, t, F, env, i, j):
    effect = int(i==j)
    R = env["R"]
    # how B changes T
    term1 = np.zeros(R)
    for k in range(1, R+1):
        ind = dep_order.index(k)
        B_list = np.exp(G[:, :ind+1]@t[:ind+1]) # every bug's growth by Rk depletion
        g_list = G[:, ind] # every bug's growth rate by Rk depletion
        term1[k-1] = 1 / ( B_list[G[:, ind]==g[:, k-1]] @ g_list[G[:, ind]==g[:, k-1]] ) # only consider those bugs eating Rk
    term1 = term1 * (-F[:, i])
    # how T changes another B
    term2 = np.zeros(R)
    for k in range(1, R):
        term2[dep_order[k-1]-1] = G[j, k-1] - G[j, k]
    term2[dep_order[-1]-1] = G[j, R-1]
    effect += term1@term2
    # print(i, j, term1, term2)
    return effect

def Pert_mat(g, dep_order, G, t, F, env):
    N = env["N"]
    P = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            P[i, j] = b_to_b(g, dep_order, G, t, F, env, i, j)
    return P

# dynamical stability, for the co-utilizers
def b_to_b_co(g, x, dep_order, G, t, F, env, i, j):
    effect = int(i==j)
    R = env["R"]
    # how B changes T
    term1 = np.zeros(R)
    for k in range(1, R+1):
        ind = dep_order.index(k)
        B_list = np.exp(G[:, :ind+1]@t[:ind+1]) # every bug's growth by Rk depletion
        g_list = G[:, ind] # every bug's growth rate by Rk depletion
        available_resources = np.array(dep_order[ind:])-1
        coeffs = x[:, k-1]*g[:, k-1] / np.sum(g[:, available_resources]*x[:, available_resources], axis=1)
        term1[k-1] = 1 / ( (coeffs*B_list) @ g_list ) # everyone is eating Rk
    term1 = term1 * (-F[:, i])
    # how T changes another B
    term2 = np.zeros(R)
    for k in range(1, R):
        term2[dep_order[k-1]-1] = G[j, k-1] - G[j, k]
    term2[dep_order[-1]-1] = G[j, R-1]
    effect += term1@term2
    # print(i, j, term1, term2)
    return effect
def Pert_mat_co(g, x, dep_order, G, t, F, env):
    N = env["N"]
    P = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            P[i, j] = b_to_b_co(g, x, dep_order, G, t, F, env, i, j)
    return P

# for plotting the cloud diagram
def ticking(xlo, xhi):
    xticks= []
    xticklabels=[]
    x1, x2 = floor(xlo), ceil(xhi)
    for i in range(x1, x2):
        xticks.extend([log10(j*(10**i)) for j in range(1, 10)])
        xticklabels.append(str(10**i))
        xticklabels.extend(["" for j in range(8)])
    xticks.append(x2)
    xticklabels.append(str(10**x2))
    xticklabels = [xticklabels[idx] for idx, i in enumerate(xticks) if xlo<=i<=xhi]
    xticks = [i for i in xticks if xlo<=i<=xhi]
    return xticks, xticklabels


###################################################################################################
# everything related to terry hwa's growth theory
def G_mat_diaux_hwa(g, gC, pref, dep_order, rho):
    '''
    g, pref: array([N, R])
    gC: float
    dep_order: array([R])
    rho: array([N])
    '''
    N, R = g.shape
    G = np.zeros([N, R])
    for i_n in range(N):
        for i_t in range(R):
            for i in range(R):
                top_resource = pref[i_n][i]
                if(top_resource in dep_order[i_t:]):
                    break
            coeff = (rho[i_n]+(1-rho[i_n])*R)
            gtilde = g[i_n, top_resource-1]*coeff
            G[i_n, i_t] = 1/(1/gtilde + 1/gC)
    return G
def G_mat_cout_hwa(g, gC, dep_order, rho):
    '''
    g: array([N, R])
    gC: float
    dep_order: array([R])
    rho: array([N])
    '''
    N, R = g.shape
    G = np.zeros([N, R])
    for i_n in range(N):
        for i_t in range(R):
            present_res = dep_order[i_t:]-1
            coeff = (rho[i_n]*len(present_res)+(1-rho[i_n])*R) / len(present_res)
            gtilde = np.sum(g[i_n, present_res])*coeff
            G[i_n, i_t] = 1/(1/gtilde + 1/gC)
    return G

def F_mat_diaux_hwa(g, gC, pref, G, dep_order, logD, rho):
    '''
    g, pref, G: array([N, R])
    gC, logD: float
    dep_order: array([R])
    rho: array([N])
    '''
    N, R = g.shape
    F_mat = np.zeros([R, N])
    t = np.linalg.inv(G)@np.ones(R)*logD
    # because the g in the input is actually g_enz, we need to convert them to g_real to use as growth rates
    rho_expand = np.tile(rho, (R, 1))
    g_real = 1/(1/(g*(rho_expand+(1-rho_expand)*R))+1/gC)
    for i_n in range(N):
        coeff = 1
        start = 0
        for i_r in range(R):
            if(start < R and pref[i_n][i_r] in dep_order[start:]):
                end = dep_order.index(pref[i_n][i_r]) + 1
                delta_t = sum(t[start:end])
                g_temp = g_real[i_n, pref[i_n][i_r]-1]
                F_mat[pref[i_n][i_r]-1, i_n] = coeff * (exp(g_temp*delta_t) - 1)
                coeff *= exp(g_temp*delta_t)
                start = end
            else:
                continue
    return F_mat
def F_mat_cout_hwa(g, G, dep_order, logD):
    '''
    g, G: array([N, R])
    logD: float
    dep_order: array([R])
    '''
    N, R = g.shape
    F_mat = np.zeros([R, N])
    t = np.linalg.inv(G)@np.ones(R)*logD
    for i_n in range(N):
        coeff = 1
        for i_t in range(R):
            present_res = dep_order[i_t:]-1
            for r in present_res:
                F_mat[r][i_n] += coeff * (( g[i_n, r] / np.sum(g[i_n, present_res]) ) * (exp(G[i_n, i_t]*t[i_t]) - 1) )
            coeff *= exp(G[i_n, i_t]*t[i_t])
    return F_mat

def b_to_b_diaux_hwa(g, gC, dep_order, G, t, F, env, i, j, rho):
    '''
    g, G: array([N, R])
    gC: float
    dep_order, t: array([R])
    F: array([R, N])
    env: dict
    rho: array([N])
    '''
    # because the g in the input is actually g_enz, we need to convert them to g_real to use as growth rates
    R = env["R"]
    rho_expand = np.tile(rho, (R, 1))
    g_real = 1/(1/(g*(rho_expand+(1-rho_expand)*R))+1/gC)
    effect = int(i==j)
    # how B changes T
    term1 = np.zeros(R)
    for k in range(1, R+1):
        ind = dep_order.index(k)
        B_list = np.exp(G[:, :ind+1]@t[:ind+1]) # every bug's growth by Rk depletion
        g_list = G[:, ind] # every bug's growth rate by Rk depletion
        term1[k-1] = 1 / ( B_list[G[:, ind]==g_real[:, k-1]] @ g_list[G[:, ind]==g_real[:, k-1]] ) # only consider those bugs eating Rk
    term1 = term1 * (-F[:, i])
    # how T changes another B
    term2 = np.zeros(R)
    for k in range(1, R):
        term2[dep_order[k-1]-1] = G[j, k-1] - G[j, k]
    term2[dep_order[-1]-1] = G[j, R-1]
    effect += term1@term2
    # print(i, j, term1, term2)
    return effect
def Pert_mat_diaux_hwa(g, gC, dep_order, G, t, F, env, rho):
    N = env["N"]
    P = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            P[i, j] = b_to_b_diaux_hwa(g, gC, dep_order, G, t, F, env, i, j, rho)
    return P
def b_to_b_cout_hwa(g, dep_order, G, t, F, env, i, j):
    '''
    g, G: array([N, R])
    dep_order: list(R)
    t: array([R])
    F: array([R, N])
    env: dict
    '''
    effect = int(i==j)
    R = env["R"]
    # how B changes T
    term1 = np.zeros(R)
    for k in range(1, R+1):
        ind = dep_order.index(k)
        B_list = np.exp(G[:, :ind+1]@t[:ind+1]) # every bug's growth by Rk depletion
        g_list = G[:, ind] # every bug's growth rate by Rk depletion
        available_resources = np.array(dep_order[ind:])-1
        coeffs = g[:, k-1] / np.sum(g[:, available_resources], axis=1)
        term1[k-1] = 1 / ( (coeffs*B_list) @ g_list ) # everyone is eating Rk
    term1 = term1 * (-F[:, i])
    # how T changes another B
    term2 = np.zeros(R)
    for k in range(1, R):
        term2[dep_order[k-1]-1] = G[j, k-1] - G[j, k]
    term2[dep_order[-1]-1] = G[j, R-1]
    effect += term1@term2
    # print(i, j, term1, term2)
    return effect
def Pert_mat_cout_hwa(g, dep_order, G, t, F, env):
    N = env["N"]
    P = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            P[i, j] = b_to_b_cout_hwa(g, dep_order, G, t, F, env, i, j)
    return P








##################################################################################################################
# lag related functions
# what resource is the species eating
def resource_eating(pref_list, dep_order):
    N, R = pref_list.shape[0], pref_list.shape[1]
    RE = np.zeros([N, R], dtype=np.int32)
    for i_n in range(N):
        for i_t in range(R):
            for i in range(R):
                top_resource = pref_list[i_n][i]
                if(top_resource in dep_order[i_t:]):
                    RE[i_n, i_t] = top_resource
                    break
    return RE

#def a state matrix
def niche_growth_state(RE, taus, t_order):
    N, R = taus.shape[0], taus.shape[1]
    # here we define a niche growth state matrix S, whose elements can be 0, 1, or 2
    # 0 means no growth; 1 means growth during full niche; 2 means part of of niche is taken by lag
    # and define a tau_mod - the lag time in each niche for those 2 elements in S. 
    S = np.ones([N, R])
    tau_mod = np.zeros([N, R])
    for species in range(N):
        # check the behavior between rsi-th t-niche and rei-th t-niche
        # which starts from 1 (second t-niche),
        # because we assume sum(t-niche)<<T, which means the lag in first t-niche is due to recovering from dormancy,
        # and we assume them to be species- and resource- independent. 
        res_last = 0 
        in_lagphase = 0
        tau_new = 0
        for niche in np.arange(1, R):
            # if next time niche is a different resource, 
            # we assume the species would renew the current lag by the lagtime of
            # switching from the previous (actively consuming) nutrient to the current nutrient
            # regardless of whether it was already in the middle of a lag phase or not. 
            if(RE[species][niche]!=RE[species][niche-1]): 
                tau_new = taus[species, RE[species][res_last]-1, RE[species][niche]-1]
                if(tau_new < t_order[niche]):
                    S[species, niche] = 2
                    tau_mod[species, niche] = tau_new
                    res_last = niche
                    in_lagphase = 0
                else:
                    S[species, niche] = 0
                    tau_new -= max(0, t_order[niche])
                    in_lagphase = 1
            # if the next time niche is a same resource, 
            # when the species has not yet finished a lag, it would continue being in that lag
            # with no need to renew the lagtime value. 
            elif(in_lagphase == 1):
                if(tau_new < t_order[niche]):
                    S[species, niche] = 2
                    tau_mod[species, niche] = tau_new
                    in_lagphase = 0
                    res_last = niche
                else:
                    S[species, niche] = 0
                    tau_new -= max(0, t_order[niche])
    return S, tau_mod

# this gives the first iteration of the tau_mod
def tau_gtoG(taus, pref, dep_order):
    N, R = taus.shape[0], taus.shape[1]
    tau_G = np.zeros([N, R])
    for i_n in range(N):
        top_resource_prev = -1
        for i_t in range(R):
            for i in range(R):
                top_resource = pref[i_n][i]
                if(top_resource in dep_order[i_t:]):
                    break
            if(i_t>=1):
                tau_G[i_n, i_t] = taus[i_n, top_resource_prev-1, top_resource-1]
            top_resource_prev = top_resource
    return tau_G

# define a function to do the iterative thing
def tsolve_iter(G, RE, taus, tau_mod, logD):
    # species-niche growth state S
    N, R = taus.shape[0], taus.shape[1]
    S = np.ones([N, R])
    # make the modifications in the first iteration
    converged = 0
    t_iter_compare = np.zeros(R)
    for count in range(10):
        rhs = logD + np.diag( (G*(S>0)) @np.transpose(tau_mod))
        if(np.linalg.matrix_rank(G*(S>0))>=N):
        # if(1):
            t_iter = np.linalg.inv(G*(S>0))@rhs
            S, tau_mod = niche_growth_state(RE, taus, t_iter)
            if ((t_iter_compare==t_iter).all() and np.sum(t_iter)<=24):
                converged = 1
                break
            t_iter_compare = t_iter
        else:
            converged = 0
            break
    return converged, t_iter, S, tau_mod

def F_mat_lag(g, gC, pref_list, dep_order, rho, logD, N, R, taus):
    RE = resource_eating(pref_list, dep_order)
    tau_mod = tau_gtoG(taus, pref_list, dep_order)
    # G = G_mat(g, pref_list, dep_order, N, R)
    G = G_mat_diaux_hwa(g, gC, pref_list, dep_order, rho)
    _, t_iter, S, tau_mod = tsolve_iter(G, RE, taus, tau_mod, logD)
    F_mat = np.zeros([R, N])
    G_mod = G*(S>0)
    for species in range(N):
        current_growth = G_mod[species][0]*(t_iter[0]-tau_mod[species][0])
        coeff = 1
        for niche in range(1, R):
            if(RE[species, niche-1]!=RE[species, niche]):
                F_mat[species, RE[species, niche-1]-1] = coeff * (exp(current_growth) - 1)
                coeff *= exp(current_growth)
                current_growth = G_mod[species][niche]*(t_iter[niche]-tau_mod[species][niche])
            else:
                current_growth += G_mod[species][niche]*(t_iter[niche]-tau_mod[species][niche])
            if(niche==R-1):
                F_mat[species, RE[species, niche]-1] = coeff * (exp(current_growth) - 1)
    return np.transpose(F_mat)

def b_to_b_hwa_lag(g, gC, dep_order, G, t, F, S, tau_mod, env, rho, i, j):
    effect = int(i==j)
    R = env["R"]
    N = env["N"]
    rho_expand = np.tile(rho, (R, 1))
    g_real = 1/(1/(g*(rho_expand+(1-rho_expand)*R))+1/gC)
    G = G*(S>0)
    # how B changes T
    term1 = np.zeros(R)
    for k in range(1, R+1):
        ind = dep_order.index(k)
        t_mod = np.tile(t[:ind+1], [N, 1]) - tau_mod[:, :ind+1]
        B_list = np.exp(np.diag(G[:, :ind+1]@(t_mod.T))) # every bug's growth by Rk depletion
        g_list = G[:, ind] # every bug's growth rate by Rk depletion
        if( B_list[G[:, ind]==g_real[:, k-1]] @ g_list[G[:, ind]==g_real[:, k-1]] > 0):
            term1[k-1] = 1 / ( B_list[G[:, ind]==g_real[:, k-1]] @ g_list[G[:, ind]==g_real[:, k-1]] ) # only consider those bugs eating Rk
    term1 = term1 * (-F[:, i])
    # how T changes another B
    term2 = np.zeros(R)
    for k in range(1, R):
        term2[dep_order[k-1]-1] = G[j, k-1] - G[j, k]
    term2[dep_order[-1]-1] = G[j, R-1]
    effect += term1@term2
    return effect
def Pert_mat_hwa_lag(g, gC, dep_order, G, t, F, S, tau_mod, rho, env):
    N = env["N"]
    P = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            P[i, j] = b_to_b_hwa_lag(g, gC, dep_order, G, t, F, S, tau_mod, env, rho, i, j)
    return P