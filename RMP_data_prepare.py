# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:33:15 2023

@author: user
"""

import pyuda
import numpy as np
import matplotlib.pyplot as plt

def interpolate_gaps(values, limit=None):
    """
    Fill gaps using linear interpolation, optionally only fill gaps up to a
    size of `limit`.
    """
    values = np.asarray(values)
    i = np.arange(values.size)
    valid = np.isfinite(values)
    filled = np.interp(i, i[valid], values[valid])

    if limit is not None:
        invalid = ~valid
        for n in range(1, limit+1):
            invalid[:-n] &= invalid[n:]
        filled[invalid] = np.nan

    return filled


# Create a client instance
client = pyuda.Client()
shotnum = 48312

q95 = client.get('/EPM/OUTPUT/GLOBALPARAMETERS/Q95', shotnum)

#print(q95.description)
print(q95.dims)
#print(q95.dims[1].description) there is only time dimension, 1 dimension
att_q = dir(q95)
print(q95.shape)
#print(att_q)
#print(q95.units)  no units

print(q95.dims[0].units) 
#print(q95.dims[1].data) there is only 1 dimension
# print(q95.data)
# print(q95.errors) no error is calculated



new_q95 = interpolate_gaps(q95.data, limit=None)*(-1)
# print(new_q95)


P_density = client.get('/AYC/N_E_CORE', shotnum)
print(P_density.dims)
#print(P_density.description) does not show description
att_v = dir(P_density)
#print(att_v)
print(P_density.shape)
print(P_density.units)
print(P_density.dims[0].units) 
#print(P_density.dims[1].units) does not show units
# print(P_density.errors) no error is calculated

new_core_ne = interpolate_gaps(P_density.data, limit=None)

D_alpha = client.get('/XIM/DA/HM10/T', shotnum)
# print(D_alpha.description)  does not show description
# att_d = dir(D_alpha)
# print(att_d)
print(D_alpha.shape)
print(D_alpha.units)
print(D_alpha.dims[0].units)
# print(D_alpha.errors) no error is calculated


mx_q95 = max(q95.dims[0].data)
print(mx_q95)

mx_pd = max(P_density.dims[0].data)
print(mx_pd)

mx_da = max(D_alpha.dims[0].data)
print(mx_da)

pick = min(mx_q95, mx_pd, mx_da)

iq_list = []

for iq in q95.dims[0].data:
    if iq >= pick:
        iq_list.append(np.where(q95.dims[0].data == iq)[0][0])
ind_q = min(iq_list)    
print(min(iq_list))    
        
ipd_list = []

for ipd in P_density.dims[0].data:
    if ipd >= pick:
        ipd_list.append(np.where(P_density.dims[0].data == ipd)[0][0])      

ind_pd = min(ipd_list)
print(min(ipd_list))  
    
ida_list = []

for ida in D_alpha.dims[0].data:
    if ida >= pick:
        ida_list.append(np.where(D_alpha.dims[0].data == ida)[0][0])


ind_da = min(ida_list)
print(min(ida_list))




plt.figure(figsize=(9,6))
plt.subplot(3,1,1)
plt.plot(q95.dims[0].data[:ind_q], new_q95[:ind_q],'o', label= 'q95')
plt.title('time trace for discharge {}'.format(shotnum))
plt.legend()


plt.subplot(3,1,2)
plt.plot(P_density.dims[0].data[:ind_pd], new_core_ne[:ind_pd],'o', label= 'density')

plt.legend()

plt.subplot(3,1,3)
plt.plot(D_alpha.dims[0].data[:ind_da], D_alpha.data[:ind_da],'o', label= 'D-alpha')
plt.xlabel('time: second')


plt.legend()
plt.show()






