# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# MUSCAT data reduction. Results analysis
# Beam shape analysis. This script generates the Figs. 
#
# Marcial Becerril, @ 06 May 2024
# Latest Revision: 06 May 2024
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #


import sys, getopt

import argparse

import os as os
import numpy as np

from matplotlib.pyplot import *
ion()
from matplotlib.patches import Ellipse

from scipy import signal
from scipy.signal import find_peaks

from astropy.io import fits

# LaTEX format
#rc('text', usetex=True)
rc('font', family='serif', size='20')
rc('font', weight='bold')

import sys
sys.path.append('../misc')

from msg_custom import *


# DIRECTORIES
# -------------------------------------------------------
DET_POS_PATH = "../src/detector_positions_per_channel/"


# Read all the beam positions
# -------------------------------------------------------
# Channels available
chns = ['clones', 'sith', 'empire', 'jedi']

chn_array = {
	"clones" : "Array 2",
	"sith" : "Array 3",
	"empire" : "Array 5",
	"jedi" : "Array 6"
}

# Obs Nums to analyse
obs_nums = ['094468', '094469', '094471', '098811', '099077', '099087', 
			'099108', '100280', '100286', '100292', '100781', '100785',
			'100789', '100908']

# Flag para el beam shape
flag_obs_beam = ['099087', '100292', '100781', '100785', '100789', '100908']

# Get colors
norm = matplotlib.colors.Normalize(vmin=0, vmax=len(obs_nums))

dets_position = {}
err_amp = {}

i = 0
for obs in obs_nums:
	rgba_color = cm.gist_rainbow(norm(i), bytes=False) 
	
	for chn in chns:
		# Detectors position
		if not chn in dets_position:
			dets_position[chn] = {}
			err_amp[chn] = {}

		path = DET_POS_PATH+chn+'/beam_map-'+chn+'-'+obs+'.npy'
		path_err = DET_POS_PATH+chn+'/err_weight-'+chn+'-'+obs+'.npy'

		beam_data = np.load(path, allow_pickle=True).item()
		try:
			weights_data = np.load(path_err)
			amp_err = weights_data
		except:
			pass

		# Plot the position
		az = beam_data[chn]['az']
		el = beam_data[chn]['el']
		fwhm_a = beam_data[chn]['fwhm_a']
		fwhm_b = beam_data[chn]['fwhm_b']
		# Amplitude
		amp = beam_data[chn]['A']

		flag = beam_data[chn]['bad']

		for k in range(len(az)):
			kid = 'K'+str(k).zfill(3)
			if not kid in dets_position[chn]:
				dets_position[chn][kid] = {}
				dets_position[chn][kid]['az'] = []
				dets_position[chn][kid]['el'] = []
				dets_position[chn][kid]['fwhm_a'] = []
				dets_position[chn][kid]['fwhm_b'] = []
				dets_position[chn][kid]['A'] = []
				dets_position[chn][kid]['A_errs'] = []

			if not flag[k]:
				dets_position[chn][kid]['az'].append(az[k])
				dets_position[chn][kid]['el'].append(el[k])

				if not obs in flag_obs_beam:
					dets_position[chn][kid]['fwhm_a'].append(np.max([fwhm_a[k], fwhm_b[k]]))
					dets_position[chn][kid]['fwhm_b'].append(np.min([fwhm_a[k], fwhm_b[k]]))

	i += 1


# Get the definitive list of positions
# -------------------------------------------------------
# This section generates the Figure 8.

# Number of threshold points
n_thresh = 3

cnt_det = 0

chn_final_pos = {}

mean_new_az_s = []
mean_new_el_s = []

fig, ax = subplots(1, 1, sharex=True, sharey=True)
#subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.08, hspace=0.0, wspace=0.0)

chns_det = {}
for chn in chns:
	chns_det[chn] = {}
	chn_final_pos[chn] = {}
	for kid in dets_position[chn].keys():
		chn_final_pos[chn][kid] = {}
		# Detector position
		az_kid = dets_position[chn][kid]['az']
		el_kid = dets_position[chn][kid]['el']
		# Beam shape
		fwhm_a_kid = dets_position[chn][kid]['fwhm_a']
		fwhm_b_kid = dets_position[chn][kid]['fwhm_b']
		# Amplitude
		a_kid = dets_position[chn][kid]['A']
		# Amplitude error
		a_kid_err = dets_position[chn][kid]['A_errs']

		# Mean and standard deviation
		if len(az_kid) > n_thresh:

			# Position
			mean_az = np.nanmean(az_kid)
			err_az = np.nanstd(az_kid)
			mean_el = np.nanmean(el_kid)
			err_el = np.nanstd(el_kid)

			# FWHM
			mean_fwhm_a = np.nanmean(fwhm_a_kid)
			err_fwhm_a = np.nanstd(fwhm_a_kid)
			mean_fwhm_b = np.nanmean(fwhm_b_kid)
			err_fwhm_b = np.nanstd(fwhm_b_kid)

			# Amplitude
			wa = (1/np.array(a_kid_err)**2)
			mean_a = np.sum(wa*a_kid)/np.sum(wa)

			mean_amp_err = np.sqrt(1/np.sum(wa))

			# Flagging positions
			if err_el <= 2 and err_az <= 2:

				chn_final_pos[chn][kid]['az'] = mean_az
				chn_final_pos[chn][kid]['az_err'] = err_az

				chn_final_pos[chn][kid]['el'] = mean_el
				chn_final_pos[chn][kid]['el_err'] = err_el

				chn_final_pos[chn][kid]['fwhm_a'] = mean_fwhm_a
				chn_final_pos[chn][kid]['fwhm_a_err'] = err_fwhm_a

				chn_final_pos[chn][kid]['fwhm_b'] = mean_fwhm_b
				chn_final_pos[chn][kid]['fwhm_b_err'] = err_fwhm_b

				chn_final_pos[chn][kid]['A'] = mean_a#mean_amp
				chn_final_pos[chn][kid]['A_err'] = mean_amp_err#err_new_a

				ax.plot(mean_az, mean_el, 'r.')
				ax.errorbar(mean_az, mean_el, xerr=err_az, yerr=err_el, color='k', capsize=2)

				mean_new_az_s.append(mean_az)
				mean_new_el_s.append(mean_el)

				cnt_det += 1

			else:
				dist = np.sqrt((mean_az-az_kid)**2 + (mean_el-el_kid)**2)
				
				idx_max = np.argmax(dist)

				del az_kid[idx_max], el_kid[idx_max]
				
				try:
					del fwhm_a_kid[idx_max], fwhm_b_kid[idx_max]
				except:
					pass

				try:
					del a_kid[idx_max-4]
					del a_kid_err[idx_max-4]
				except:
					pass

				mean_new_az = np.mean(az_kid)
				mean_new_el = np.mean(el_kid)

				err_new_az = np.std(az_kid)
				err_new_el = np.std(el_kid)

				mean_new_fwhm_a = np.mean(fwhm_a_kid)
				mean_new_fwhm_b = np.mean(fwhm_b_kid)

				err_new_fwhm_a = np.std(fwhm_a_kid)
				err_new_fwhm_b = np.std(fwhm_b_kid)

				wa_n = (1/np.array(a_kid_err)**2)
				mean_new_a = np.sum(wa_n*a_kid)/np.sum(wa_n)

				mean_amp_new_err = np.sqrt(1/np.sum(wa_n))
		
				chn_final_pos[chn][kid]['az'] = mean_new_az
				chn_final_pos[chn][kid]['az_err'] = err_new_az

				chn_final_pos[chn][kid]['el'] = mean_new_el
				chn_final_pos[chn][kid]['el_err'] = err_new_el

				chn_final_pos[chn][kid]['fwhm_a'] = mean_new_fwhm_a
				chn_final_pos[chn][kid]['fwhm_a_err'] = err_new_fwhm_a

				chn_final_pos[chn][kid]['fwhm_b'] = mean_new_fwhm_b
				chn_final_pos[chn][kid]['fwhm_b_err'] = err_new_fwhm_b

				chn_final_pos[chn][kid]['A'] = mean_new_a
				chn_final_pos[chn][kid]['A_err'] = mean_amp_new_err 

				ax.plot(mean_new_az, mean_new_el, 'r.')
				ax.errorbar(mean_new_az, mean_new_el, xerr=err_new_az, yerr=err_new_el, color='k', capsize=2)

				mean_new_az_s.append(mean_new_az)
				mean_new_el_s.append(mean_new_el)

				cnt_det += 1

	chns_det[chn] = cnt_det
	print(chn)
	print('Number of detectors: ', cnt_det)
	cnt_det = 0


ax.axis('equal')
ax.set_xlabel('[arcsec]')
ax.set_ylabel('[arcsec]')

ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)


# Get the histograms of positions uncertainties.
# -------------------------------------------------------
# This section generates the Figure 9.

fig, axs = subplots(2, 2, sharex=True, sharey=True)
subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.08, hspace=0.0, wspace=0.0)

chn_err = {}

for c, chn in enumerate(chns):
	chn_err[chn] = {}
	chn_err[chn]['az'] = []
	chn_err[chn]['el'] = []
	for kid in chn_final_pos[chn].keys():
		if 'az' in chn_final_pos[chn][kid]:
			chn_err[chn]['az'].append(chn_final_pos[chn][kid]['az_err'])
			chn_err[chn]['el'].append(chn_final_pos[chn][kid]['el_err'])

	i = c%2
	j = int(c/2)

	print(chn)
	print('AZ_err median:', np.median(chn_err[chn]['az']) )
	print('AZ_err mean:', np.mean(chn_err[chn]['az']) )
	print('EL_err median:', np.median(chn_err[chn]['el']) )
	print('EL_err mean:', np.mean(chn_err[chn]['el']) )

	axs[j, i].hist(chn_err[chn]['az'], histtype='stepfilled', ec='k', lw=2, bins=np.linspace(0, 2, 40), color='r', alpha=0.75, label='AZ')
	axs[j, i].hist(chn_err[chn]['el'], histtype='stepfilled', ec='k', lw=2, bins=np.linspace(0, 2, 40), color='b', alpha=0.75, label='EL')

	axs[j, i].text(1.5, 25, chn_array[chn])


axs[0,1].legend(loc='center right')

axs[1,0].set_xlabel(r'Error [mm]')
axs[1,1].set_xlabel(r'Error [mm]')

axs[1,0].set_ylabel(r'# detectors')
axs[0,0].set_ylabel(r'# detectors')



# Plot the heat maps of error positions
# -------------------------------------------------------
# These graphs are not presented in the paper.

radius = 2.5
alpha = 0.85

max_az = 1.6
min_az = 0

max_el = 1.6
min_el = 0

hm = cm.get_cmap('gist_rainbow', 256)

fig_az, ax_az = subplots()
ax_az.axis('equal')

fig_el, ax_el = subplots()
ax_el.axis('equal')

for chn in chns:
	for kid in chn_final_pos[chn].keys():
		if 'az' in chn_final_pos[chn][kid]:
			mean_az = chn_final_pos[chn][kid]['az']
			mean_el = chn_final_pos[chn][kid]['el']

			err_az = chn_final_pos[chn][kid]['az_err']
			err_el = chn_final_pos[chn][kid]['el_err']

			color_az = int(256*( err_az - min_az )/( max_az - min_az ) )
			color_el = int(256*( err_el - min_el )/( max_el - min_el ) )

			pix_az = Circle((mean_az, mean_el), radius, color=hm(color_az), alpha=alpha, ec='k', lw=1.5)
			pix_el = Circle((mean_az, mean_el), radius, color=hm(color_el), alpha=alpha, ec='k', lw=1.5)

			ax_az.add_artist(pix_az)
			ax_el.add_artist(pix_el)

ax_az.set_xlabel('[arcsec]')
ax_az.set_ylabel('[arcsec]')

ax_el.set_xlabel('[arcsec]')
ax_el.set_ylabel('[arcsec]')

values = np.linspace(min_az, max_el, 256)
cax = ax_az.scatter(values, values*0, c=values, cmap='gist_rainbow')
cax.set_visible(False)
cbar = fig_az.colorbar(cax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.ax.set_title(r'Error Azimuth [arcsec]')

values = np.linspace(min_el, max_el, 256)
cax = ax_el.scatter(values, values*0, c=values, cmap='gist_rainbow')
cax.set_visible(False)
cbar = fig_el.colorbar(cax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.ax.set_title(r'Error Elevation [arcsec]')



# Plot the heat maps of beam shapes
# -------------------------------------------------------
# This section generates the Figures 12 and 13

radius = 2.5
alpha = 0.85

max_ecc = 0.75
min_ecc = 0.55

max_width = 6.6
min_width = 5.6

hm = cm.get_cmap('gist_rainbow', 256)

fig_ecc, ax_ecc = subplots()
ax_ecc.axis('equal')

fig_width, ax_width = subplots()
ax_width.axis('equal')

for chn in chns:
	for kid in chn_final_pos[chn].keys():
		if 'az' in chn_final_pos[chn][kid]:

			mean_az = chn_final_pos[chn][kid]['az']
			mean_el = chn_final_pos[chn][kid]['el']

			mean_fwhm_a = chn_final_pos[chn][kid]['fwhm_a']
			mean_fwhm_b = chn_final_pos[chn][kid]['fwhm_b']

			a = np.max([mean_fwhm_a, mean_fwhm_b])
			b = np.min([mean_fwhm_a, mean_fwhm_b])

			ecc = np.sqrt(1-b**2/a**2)
			e = (a+b)/2

			color_ecc = int(256*( ecc - min_ecc )/( max_ecc - min_ecc ) )
			color_width = int(256*( e - min_width )/( max_width - min_width) )

			pix_ecc = Circle((mean_az, mean_el), radius, color=hm(color_ecc), alpha=alpha, ec='k', lw=1.5)
			ellipse = Ellipse(xy=(mean_az, mean_el), width=a, height=b, angle=0, color=hm(color_width), alpha=alpha, ec='k', lw=1.5)
			
			ax_ecc.add_patch(pix_ecc)
			ax_width.add_patch(ellipse)

			ax_ecc.set_xlabel("[arcsec]")
			ax_ecc.set_ylabel("[arcsec]")

			ax_width.set_xlabel("[arcsec]")
			ax_width.set_ylabel("[arcsec]")


values = np.linspace(min_ecc, max_ecc, 256)
cax = ax_ecc.scatter(values, values*0, c=values, cmap='gist_rainbow')
cax.set_visible(False)
cbar = fig_ecc.colorbar(cax, orientation='vertical', fraction=0.046, pad=0.08)
cbar.set_label('Eccentricity', labelpad=-140)

values = np.linspace(min_width, max_width, 256)
cax = ax_width.scatter(values, values*0, c=values, cmap='gist_rainbow')
cax.set_visible(False)
cbar = fig_width.colorbar(cax, orientation='vertical', fraction=0.046, pad=0.08)
cbar.set_label('Mean axis ["]', labelpad=-120)


# Beam shape histograms
# -------------------------------------------------------
# This section generates the Figure 11.

fig, axs = subplots(2, 2, sharex=True, sharey=True)
subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.08, hspace=0.0, wspace=0.0)

chn_beam = {}
for c, chn in enumerate(chns):
	chn_beam[chn] = {}
	chn_beam[chn]['fwhm_a'] = []
	chn_beam[chn]['fwhm_b'] = []
	for kid in chn_final_pos[chn].keys():
		if 'az' in chn_final_pos[chn][kid]:
			chn_beam[chn]['fwhm_a'].append(chn_final_pos[chn][kid]['fwhm_a'])
			chn_beam[chn]['fwhm_b'].append(chn_final_pos[chn][kid]['fwhm_b'])

	i = c%2
	j = int(c/2)

	print(chn)
	print('FWHM-A median:', np.median(chn_beam[chn]['fwhm_a']) )
	print('FWHM-A mean:', np.mean(chn_beam[chn]['fwhm_a']) )
	print('FWHM-B median:', np.median(chn_beam[chn]['fwhm_b']) )
	print('FWHM-B mean:', np.mean(chn_beam[chn]['fwhm_b']) )

	axs[j, i].hist(chn_beam[chn]['fwhm_a'], histtype='stepfilled', bins=np.linspace(4.2, 8.2, 40), lw=2, color='r', ec='k', alpha=0.75, label='FWHM-a')
	axs[j, i].hist(chn_beam[chn]['fwhm_b'], histtype='stepfilled', bins=np.linspace(4.2, 8.2, 40), lw=2, color='b', ec='k', alpha=0.75, label='FWHM-b')
	
	axs[j, i].text(7.5, 20, chn_array[chn])


axs[0,1].legend(loc='best')

axs[1,0].set_xlabel(r'FWHM [arcsec]')
axs[1,1].set_xlabel(r'FWHM [arcsec]')

axs[1,0].set_ylabel(r'# detectors')
axs[0,0].set_ylabel(r'# detectors')

