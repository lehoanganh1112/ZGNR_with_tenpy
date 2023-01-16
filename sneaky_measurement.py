'''
This script is for measuring quantities of MPS after each sweep
'''

import tenpy
import numpy as np
import h5py

Lx = 121
Ly = 2
dope = 12
Nsites = int((Lx-1)*Ly*2 + Ly)

results = tenpy.tools.hdf5_io.load('dmrg_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_dope'+str(dope)+'_clean.h5')
psi_test = results['psi']
psi_test.canonical_form()

sz = psi_test.expectation_value('Sz')
np.savetxt('sz_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_dope'+str(dope)+'_clean.txt', sz)


ee = psi_test.entanglement_entropy()
np.savetxt('ee_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_dope'+str(dope)+'_clean.txt', ee)

ntot = psi_test.expectation_value('Ntot')
np.savetxt('ntot_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_dope'+str(dope)+'_clean.txt' ,ntot)