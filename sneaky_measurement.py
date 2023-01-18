'''
This script is for measuring quantities of MPS after each sweep
'''

import tenpy
import numpy as np
import h5py

Lx = 31
Ly = 2
dope = 0
gamma = 0.5

results = tenpy.tools.hdf5_io.load('dmrg_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_dope'+str(dope)+'_clean.h5')
psi_test = results['psi']
psi_test.canonical_form()

sz = psi_test.expectation_value('Sz')
ntot = psi_test.expectation_value('Ntot')
ee = psi_test.entanglement_entropy()

# # Clean
# np.savetxt('sz_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_dope'+str(dope)+'_clean.txt', sz)
# np.savetxt('ee_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_dope'+str(dope)+'_clean.txt', ee)
# np.savetxt('ntot_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_dope'+str(dope)+'_clean.txt' ,ntot)

# Disorder
np.savetxt('sz_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_dope'+str(dope)+'_disorder.txt', sz)
np.savetxt('ee_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_dope'+str(dope)+'_disorder.txt', ee)
np.savetxt('ntot_L'+str(int(Lx-1))+'W'+str(int(Ly*2))+'_dope'+str(dope)+'_disorder.txt' ,ntot)