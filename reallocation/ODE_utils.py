import numpy as np
import os
from math import *
import random
import itertools
from scipy.optimize import root_scalar
from scipy import stats
from scipy import special
import copy
import time
import matplotlib.pyplot as plt
import collections
import pickle
from sklearn.linear_model import LinearRegression
import matplotlib.gridspec as gridspec
from utils import *

# simulation is done for a certain system, and the detailed information about the last dilution cycle is always recorded
'''
details = {"ts":[float (timepoint) for each event in this cycle], 
        "cs":[[float for max # of resource] for each event], 
        "bs":[[float for # of bugs alive] for each event]}
'''

DIEOUT_BOUND = 1e-8
INVADE_BOLUS = 1e-6

# in principle the SeqUt and CoUt class should be both inherited from a base class but I'm too lazy now
class SeqUt: # sequential utilizer
    def __init__(self, growth_rate_list_single, pref_list, biomass, id):
        '''
        growth_Rate_list_single: float array[N_res], generated from gaussian
        pref_list: int array[N_res]
        biomass: float
        '''
        self.cat = "Seq"
        self.id = id
        self.alive = True
        self.nr = len(growth_rate_list_single)
        self.g = growth_rate_list_single # g here is already converted from gaussian and by Terry's physiology
        self.pref = pref_list
        self.b = biomass
        self.eating = np.array([False for i in range(self.nr)]) # default - not eating anything
        self.inlag = False
    def Dilute(self, D):
        '''
        D: float
        '''
        self.b /= D
        if(self.b<DIEOUT_BOUND):
            self.alive = False
    def GetEating(self, Rs):
        '''
        Rs: float array of all resources
        '''
        self.eating = np.array([False for i in range(self.nr)])
        for r in self.pref:
            if(Rs[r-1]>0):
                self.eating[r-1] = True
                break
    def GetGrowthRate(self):
        return self.g @ self.eating
    def GetDep(self):
        '''
        In all cases assume yield Y=1
        '''
        return self.eating.astype(float)


class CoUt: # coutilizer
    def __init__(self, growth_rate_list_single, gC, biomass, id):
        '''
        growth_Rate_list_single: float array[1, N_res]
        gC: float
        biomass: float
        '''
        self.cat = "Cout"
        self.id = id
        self.alive = True
        self.nr = len(growth_rate_list_single)
        self.g = growth_rate_list_single # g here is already converted from gaussian and by Terry's physiology
        self.gC = gC
        self.b = biomass
        self.eating = np.array([False for i in range(self.nr)]) # default - not eating anything
        self.inlag = False
    def Dilute(self, D):
        '''
        D: float, dilution factor
        '''
        self.b /= D
        if(self.b<DIEOUT_BOUND):
            self.alive = False
    def GetEating(self, Rs):
        '''
        Rs: float array of all resources
        '''
        self.eating = (Rs>0)
    def GetGrowthRate(self): # growth rate of the sepcies with non-zero resources in R_left
        g_vec = self.g[self.eating]
        if(True not in self.eating):
            return 0
        else:
            return ( ( g_vec @ ((1-g_vec/self.gC)**-1) )**-1 + 1/self.gC )**-1
    def GetDep(self):
        '''
        In all cases assume yield Y=1
        Get the fraction of each resource in the biomass gained by this co-utilizer
        '''
        g_tilde = 1/(1/self.g-1/self.gC) # \propto c_i*k_i*E_i^max as in Terry's formalism
        dep = np.zeros(self.nr)
        dep[self.eating] = g_tilde[self.eating]/np.sum(g_tilde[self.eating])
        return dep

class EcoSystem: 
    def __init__(self, species=[]):
        '''
        Rs_init: float array [N_res]
        species: list of species; species are SeqUt, CoUt etc examples
        '''
        self.res = np.array([])
        self.species = species
        for species in self.species:
            species.alive = True
        self.last_cycle = {'ids':[species.id for species in self.species], 'ts':[], 'cs':[], 'bs':[]}
    def OneCycle(self, R0, T_dilute):
        '''
        R0: float array [N_res] added in this cycle
        T_dilute: float, cutoff of dilution time
        At the beginning of each cycle, res are at the scale of 1 and species are at the scale of 1/D
        '''
        self.species = [species for species in self.species if species.alive]
        ts, cs, bs = [], [], []
        t_switch = 0
        self.res = copy.deepcopy(R0)
        nr = len(R0)
        deplete_flag = -1
        ts.append(t_switch)
        cs.append(copy.deepcopy(self.res))
        bs.append(np.array([species.b for species in self.species]))
        while t_switch < T_dilute:
            for species in self.species:
                species.GetEating(self.res)
            t_dep = T_dilute - t_switch
            for r_id, r in enumerate(self.res):
                if r>0:
                    def remain(t):
                        return r - sum([species.b * (exp(species.GetGrowthRate()*t)-1) * species.GetDep()[r_id] for species in self.species])
                    if remain(t_dep)<0:
                        # did some tests seems like brenth is the fastest
                        t_i = root_scalar(remain, bracket = [0, t_dep], method='brenth').root
                        t_dep = min(t_dep, t_i)
                        deplete_flag = r_id
            t_switch = t_switch + t_dep
            # update the system according to the t_dep
            # first update res, then b, because we need b at the previous timepoint for res change
            for r_id, r in enumerate(self.res):
                self.res[r_id] = r - sum([species.b * (exp(species.GetGrowthRate()*t_dep)-1) * species.GetDep()[r_id] for species in self.species])
                if(deplete_flag != -1):
                    self.res[deplete_flag] = 0
            deplete_flag = -1
            for species in self.species:
                species.b = species.b * exp(species.GetGrowthRate()*t_dep)
            ts.append(t_switch)
            cs.append(copy.deepcopy(self.res))
            bs.append(np.array([species.b for species in self.species]))
        self.last_cycle = {'ids':[species.id for species in self.species], 'ts':ts, 'cs': cs, 'bs': bs}
    def MoveToNext(self, D):
        '''
        D: float, dilution rate
        This does not include adding new resources
        '''
        self.res /= D
        for species in self.species:
            species.Dilute(D)
    def CheckInvade(self, invader, D):
        '''
        invader: a species
        D: float, dilution rate
        '''
        if(len(self.species) == 0):
            return True
        if(invader.id in [species.id for species in self.species]):
            return False
        ts, cs= self.last_cycle["ts"], self.last_cycle["cs"],
        growth = 0
        for idx, t_pt in enumerate(ts[:-1]):
            if(np.sum(cs[idx])>0):
                delta_t = ts[idx+1] - ts[idx]
                invader.GetEating(cs[idx])
                growth += invader.GetGrowthRate()*delta_t
        return growth>log(D)
    def Invade(self, invader):
        '''
        invader: a species
        '''
        invader.alive = True
        invader.biomass = INVADE_BOLUS
        self.species.append(invader)



# make some function about plotting the biomass across cycles
def vis_biomass(id_list, blist):
    '''
    id_list: list of list of int; each element is the ID of species that are present at the end of a cycle
    blist: list of np.array of float; each element is the abundance of species at the end of a cycle
    '''
    all_keys = set(sum(id_list, []))
    all_info_dict = {key:[] for key in all_keys}
    for cycle, ids in enumerate(id_list):
        for key in all_keys:
            all_info_dict[key].append(0)
        for idx, id in enumerate(ids):
            all_info_dict[id][-1] = blist[cycle][idx]
    for key in all_info_dict:
        plt.plot(range(len(blist)), all_info_dict[key])
    plt.xlabel("Dilution cycles")
    plt.ylabel("Species abundance")