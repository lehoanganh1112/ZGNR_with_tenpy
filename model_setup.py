########################## Import library ##########################
import numpy as np
import scipy
import tenpy 
import random
from os.path import exists

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.models.lattice import Honeycomb, IrregularLattice
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.networks.terms import OnsiteTerms, CouplingTerms, MultiCouplingTerms
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from tenpy.simulations.ground_state_search import GroundStateSearch
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
from tenpy.networks.mps import InitialStateBuilder

tenpy.tools.misc.setup_logging(to_stdout="INFO")


def matlab_round(x):
    if x - int(x) >= 0.5:
        return int(np.ceil(x))
    else:
        return int(x)

def DivisibleQ(x, b):
    if np.remainder(x, b) == 0:
        return True
    else:
        return False

class ribbon4_clean(CouplingModel, MPOModel):
    def __init__(self, dummy):
        # Geometry of ribbon: Lx = Lx-1, Ly = Ly*2
        # E.g: Lx=121 and Ly=2 means Lx=120 and Ly=4
        Lx = 61
        Ly = 2
        U = 1
        t = 1
        sites = SpinHalfFermionSite(cons_N='N', cons_Sz = None, filling=1)
        # 4) Lattice
        lat = Honeycomb(Lx, Ly, sites,
                basis=np.array(([1, 0], [0.5, 0.5 * np.sqrt(3)])),
                positions=np.array(([0,0], [1/2, 1/2 * 1/np.sqrt(3)])), bc=['open','open'], bc_MPS='finite')
        lat_fix = IrregularLattice(lat, remove=[[0,0,0],[Lx-1,1,1]])
        lat_fix.order = lat_fix.ordering('Cstyle')

        # 5) initialize CouplingModel
        # explicit_plus_hc (bool) – If True, the Hermitian conjugate of the MPO is computed at runtime, 
        # rather than saved in the MPO.
        CouplingModel.__init__(self, lat_fix)
        # 6) add terms of the Hamiltonian
        # (u is always 0 as we have only one site in the unit cell)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
        self.add_onsite(U, 0, 'NuNd')
        self.add_onsite(U, 1, 'NuNd')
        # the `plus_hc=True` adds the h.c. term
        # 7) initialize H_MPO
        MPOModel.__init__(self, lat_fix, self.calc_H_MPO())




class ribbon4_disorder(CouplingModel, MPOModel):
    def __init__(self, dummy):
        # Geometry of ribbon: Lx = Lx-1, Ly = Ly*2
        # E.g: Lx=121 and Ly=2 means Lx=120 and Ly=4
        Lx = 30
        Ly = 2
        U = 1
        t = 1
        gamma = 0.5
        imp = 0.1
        sites = SpinHalfFermionSite(cons_N='N', cons_Sz = None, filling=1)
        # 4) Lattice
        lat = Honeycomb(Lx, Ly, sites,
                basis=np.array(([1, 0], [0.5, 0.5 * np.sqrt(3)])),
                positions=np.array(([0,0], [1/2, 1/2 * 1/np.sqrt(3)])), bc=['open','open'], bc_MPS='finite')
        lat_fix = IrregularLattice(lat, remove=[[0,0,0],[Lx-1,1,1]])
        lat_fix.order = lat_fix.ordering('Cstyle')

        # 5) initialize CouplingModel
        # explicit_plus_hc (bool) – If True, the Hermitian conjugate of the MPO is computed at runtime, 
        # rather than saved in the MPO.
        CouplingModel.__init__(self, lat_fix)
        # 6) add terms of the Hamiltonian
        # (u is always 0 as we have only one site in the unit cell)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
        self.add_onsite(U, 0, 'NuNd')
        self.add_onsite(U, 1, 'NuNd')

        sites_path = 'sites_impurity_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_gamma'+str(gamma)+'.txt'
        strength_path = 'strength_impurity_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_gamma'+str(gamma)+'.txt'

        if exists(sites_path) == False and exists(strength_path) == False:
            Nsites = lat_fix.N_sites
            sites_impurity = random.sample(range(Nsites), matlab_round((Nsites)*imp))
            strength_impurity = []
            for i in range(len(sites_impurity)):
                strength_impurity.append(np.random.uniform(-gamma, gamma))
            
            np.savetxt(sites_path, sites_impurity, fmt='%i')
            np.savetxt(strength_path, strength_impurity)
            print('Disorder terms are added!')
        else:
            print('Disorder files are already existed!')

        for index, impurity_site in enumerate(np.loadtxt(sites_path, dtype='int')):
            v_j = np.loadtxt(strength_path)[index]
            self.add_onsite_term(v_j, impurity_site, 'Nu')
            self.add_onsite_term(v_j, impurity_site, 'Nd')
        # the `plus_hc=True` adds the h.c. term
        # 7) initialize H_MPO
        MPOModel.__init__(self, lat_fix, self.calc_H_MPO())


class ribbon8_clean(CouplingModel, NearestNeighborModel, MPOModel):
    def __init__(self, dummy):
        # Geometry of ribbon: Lx = Lx-2, Ly = Ly*2
        # E.g: Lx=61 and Ly=4 means Lx=59 and Ly=8
        Lx = 61
        Ly = 4
        U = 1
        t = 1
        sites = SpinHalfFermionSite(cons_N='N', cons_Sz = None, filling=1)
        # 4) Lattice
        lat = Honeycomb(Lx, Ly, sites,
                basis=np.array(([1, 0], [0.5, 0.5 * np.sqrt(3)])),
                positions=np.array(([0,0], [1/2, 1/2 * 1/np.sqrt(3)])), bc=['open','open'], bc_MPS='finite')
        lat_fix = IrregularLattice(lat, remove=[[0,0,0],[0,1,0],[1,0,0],[0,2,0],[0,0,1],[0,1,1],
                                                [Lx-1,1,1],[Lx-1,2,1],[Lx-1,2,0],[Lx-1,3,1],[Lx-1,3,0],[Lx-1,3,1],[Lx-2,3,1]])
        lat_fix.order = lat_fix.ordering('Cstyle')
        # Define order of lattice
        def order_sort(lattice_idx):
            fluctuation = 0.05
            x, y = lat_fix.position(lattice_idx)
            ceiling = np.ceil(x)-2
            if 0-fluctuation<=y<=0+fluctuation:
                row = 0
            elif 0.288-fluctuation<=y<=0.288+fluctuation:
                row = 1
            elif 0.86-fluctuation<=y<=0.86+fluctuation:
                row = 2
            elif 1.15-fluctuation<=y<=1.15+fluctuation:
                row = 3
            elif 1.73-fluctuation<=y<=1.73+fluctuation:
                row = 4
            elif 2.02-fluctuation<=y<=2.02+fluctuation:
                row = 5
            elif 2.59-fluctuation<=y<=2.59+fluctuation:
                row = 6
            elif 2.86-fluctuation<=y<=2.86+fluctuation:
                row = 7
            return int(ceiling*Ly*2 + row)
        new_order = []
        for i in range((Lx-1)*Ly*2-1):
            for element in lat_fix.order:
                if order_sort(element) == i:
                    new_order.append(element)
        lat_fix.order = np.array(new_order)
        
        # 5) initialize CouplingModel
        # explicit_plus_hc (bool) – If True, the Hermitian conjugate of the MPO is computed at runtime, 
        # rather than saved in the MPO.
        CouplingModel.__init__(self, lat_fix)
        # 6) add terms of the Hamiltonian
        # (u is always 0 as we have only one site in the unit cell)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
        self.add_onsite(U, 0, 'NuNd')
        self.add_onsite(U, 1, 'NuNd')
        # the `plus_hc=True` adds the h.c. term
        # 7) initialize H_MPO
        MPOModel.__init__(self, lat_fix, self.calc_H_MPO())


class ribbon8_disorder(CouplingModel, NearestNeighborModel, MPOModel):
    def __init__(self, dummy):
        # Geometry of ribbon: Lx = Lx-2, Ly = Ly*2
        # E.g: Lx=61 and Ly=4 means Lx=59 and Ly=8
        Lx = 61
        Ly = 4
        U = 1
        t = 1
        gamma = 0.5
        imp = 0.1
        sites = SpinHalfFermionSite(cons_N='N', cons_Sz = None, filling=1)
        # 4) Lattice
        lat = Honeycomb(Lx, Ly, sites,
                basis=np.array(([1, 0], [0.5, 0.5 * np.sqrt(3)])),
                positions=np.array(([0,0], [1/2, 1/2 * 1/np.sqrt(3)])), bc=['open','open'], bc_MPS='finite')
        lat_fix = IrregularLattice(lat, remove=[[0,0,0],[0,1,0],[1,0,0],[0,2,0],[0,0,1],[0,1,1],
                                                [Lx-1,1,1],[Lx-1,2,1],[Lx-1,2,0],[Lx-1,3,1],[Lx-1,3,0],[Lx-1,3,1],[Lx-2,3,1]])
        lat_fix.order = lat_fix.ordering('Cstyle')
        # Define order of lattice
        def order_sort(lattice_idx):
            fluctuation = 0.05
            x, y = lat_fix.position(lattice_idx)
            ceiling = np.ceil(x)-2
            if 0-fluctuation<=y<=0+fluctuation:
                row = 0
            elif 0.288-fluctuation<=y<=0.288+fluctuation:
                row = 1
            elif 0.86-fluctuation<=y<=0.86+fluctuation:
                row = 2
            elif 1.15-fluctuation<=y<=1.15+fluctuation:
                row = 3
            elif 1.73-fluctuation<=y<=1.73+fluctuation:
                row = 4
            elif 2.02-fluctuation<=y<=2.02+fluctuation:
                row = 5
            elif 2.59-fluctuation<=y<=2.59+fluctuation:
                row = 6
            elif 2.86-fluctuation<=y<=2.86+fluctuation:
                row = 7
            return int(ceiling*Ly*2 + row)
        new_order = []
        for i in range((Lx-1)*Ly*2-1):
            for element in lat_fix.order:
                if order_sort(element) == i:
                    new_order.append(element)
        lat_fix.order = np.array(new_order)
        
        # 5) initialize CouplingModel
        # explicit_plus_hc (bool) – If True, the Hermitian conjugate of the MPO is computed at runtime, 
        # rather than saved in the MPO.
        CouplingModel.__init__(self, lat_fix)
        # 6) add terms of the Hamiltonian
        # (u is always 0 as we have only one site in the unit cell)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
        self.add_onsite(U, 0, 'NuNd')
        self.add_onsite(U, 1, 'NuNd')

        sites_path = 'sites_impurity_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_gamma'+str(gamma)+'.txt'
        strength_path = 'strength_impurity_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_gamma'+str(gamma)+'.txt'

        if exists(sites_path) == False and exists(strength_path) == False:
            Nsites = lat_fix.N_sites
            sites_impurity = random.sample(range(Nsites), matlab_round((Nsites)*imp))
            strength_impurity = []
            for i in range(len(sites_impurity)):
                strength_impurity.append(np.random.uniform(-gamma, gamma))
            
            np.savetxt(sites_path, sites_impurity, fmt='%i')
            np.savetxt(strength_path, strength_impurity)
            print('Disorder terms are added!')
        else:
            print('Disorder files are already existed!')

        for index, impurity_site in enumerate(np.loadtxt(sites_path, dtype='int')):
            v_j = np.loadtxt(strength_path)[index]
            self.add_onsite_term(v_j, impurity_site, 'Nu')
            self.add_onsite_term(v_j, impurity_site, 'Nd')
        # the `plus_hc=True` adds the h.c. term
        # 7) initialize H_MPO
        MPOModel.__init__(self, lat_fix, self.calc_H_MPO())