# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# PSF analysis of MUSCAT pointing-sources coadded maps
# coadd_psf_analysis.py
#
# Marcial Becerril, @ 28 Oct 2023
# Latest Revision: 28 Oct 2023, 18:50 GMT-6
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #


import astropy.units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats

from photutils.detection import DAOStarFinder

from scipy import optimize
from scipy.optimize import curve_fit

import numpy as np

from datetime import datetime

from matplotlib.pyplot import *
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt, patches
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('../functions')
# Mapping speed functions
from map_speed_functions import *
# Import PSF functions
from psf_functions import *


# LaTEX format
#rc('text', usetex=True)
rc('font', weight='bold')
rc('font', family='serif', size='15')

ion()

# Initial arguments
# --------------------------------------------------------

# Band to analise [um]
band = "1100"

# Rough FWHM estimation
fwhm = 10

# Threshold for source detection
thresh = 100

# Channel
chns = ["clones", "sith", "empire", "jedi"]

# Directory
diry = "../data/"

# Useful extensions
ext = ["signal", "err", "snr"]

# Obs num
obs_nums = ['094468', '094469', '094471', '098811', '099077', '099087', '099108', 
			'100280', '100286', '100292', '100781', '100785', '100789', '100908']

# Flag obs
# 1.1 mm
flag_obs = {
			"clones" : 
				[],
			"sith" : 			
				[],
			"empire" : 
				[],
			"jedi" : 
				[]
			}

flag_n = True


fig, axs = subplots(3, 5)#, sharex=True)
subplots_adjust(left=0.02, right=0.90, top=0.95, bottom=0.05, hspace=0.0, wspace=0.0)

# Extract the data
n_temp = 0
for obs in obs_nums:

	ii = n_temp%5
	jj = int(n_temp/5)
	
	chn_cnt = 0
	headers, datas, weights, refs = {}, {}, {}, {}
	for chn in chns:
		try:
			if not obs in flag_obs[chn]:
				# Filename
				filename = "muscat_commissioning_a"+band+"_"+obs+"_"+chn+".fits"	
				path = diry + '/' + filename
				# Read data
				main_header, header, data = read_map(path, ext) 

				# Default refs
				post_data, refs[chn] = get_source_refs(data[ext[0]], header[ext[0]], data[ext[2]], fwhm, thresh, norm=True)
				#refs[chn] = (0, 0)

				if refs[chn] != -1:
					# Create the useful dictionaries
					headers[chn] = header[ext[0]]
					datas[chn] = post_data
					weights[chn] = data[ext[1]]

				chn_cnt += 1

			else:
				print(obs, ' flagged!')
		except Exception as e:
			print(e)
			print(obs, ' flagged!')

	try:

		print('-----------------------------------')
		print(refs)		
		print('-----------------------------------')

		refs['jedi'] = refs['empire']
		refs['sith'] = refs['empire']
		refs['clones'] = refs['empire']

		sample_map, ref, deltas, weights_map = coadd_maps(headers, datas, weights, refs)

		# Get stats from the coadd-map
		centre_x, centre_y = ref[0], ref[1]
		centre_sample_radius = 4*fwhm

		point_source_coadd = sample_map[int(centre_y-centre_sample_radius):int(centre_y+centre_sample_radius)+1,
										int(centre_x-centre_sample_radius):int(centre_x+centre_sample_radius)+1]
		xx, yy = np.meshgrid(np.linspace(0, 2*centre_sample_radius+1, 2*centre_sample_radius+1, endpoint=False),
											np.linspace(0, 2*centre_sample_radius+1, 2*centre_sample_radius+1, endpoint=False))

		mask =~ np.isnan(point_source_coadd)

		popt, pcov = fit_beam(xx, yy, point_source_coadd, 'ElGauss', mask=mask)


		# R E S U L T S
		# ------------------------------------------------
		pos_source = (popt[1], popt[2])
		print('Beam shape')
		A = popt[0]
		print('A:', popt[0])
		a_axis = popt[3]*2*np.sqrt(2*np.log(2))
		b_axis = popt[4]*2*np.sqrt(2*np.log(2))
		print('FWHM (a): ', a_axis)
		print('FWHM (b): ', b_axis)
		theta = -180*popt[5]/np.pi
		print('theta: ', theta)
		offset = popt[6]
		print('offset: ', offset)

		# Calculate the eccentricity
		c = np.sqrt((np.max([a_axis, b_axis])/2)**2 - (np.min([a_axis, b_axis])/2)**2)
		e = c/(np.max([a_axis, b_axis])/2)
		print('eccentricity: ', e)


		# ---------------------------------------------------------------------

		im = axs[jj, ii].imshow(point_source_coadd, cmap='jet', vmin=-0.01, vmax=1.0, extent=[-0.5-centre_sample_radius,centre_sample_radius+0.5,-0.5-centre_sample_radius,centre_sample_radius+0.5], origin='lower')#, extent=[-30, 30, -30, 30])
		
		mg = axs[jj, ii].contour(point_source_coadd, levels=7, linewidths=1.0, colors='k', extent=[-0.5-centre_sample_radius,centre_sample_radius+0.5,-0.5-centre_sample_radius,centre_sample_radius+0.5])
		axs[jj, ii].set_ylim(-20, 20)
		axs[jj, ii].set_xlim(-20, 20)

		axs[jj, ii].tick_params(axis="y",direction="in", length=10, pad=-22, colors='white')
		axs[jj, ii].tick_params(axis="x",direction="in", length=10, pad=-22, colors='white')

		axs[jj, ii].xaxis.set_tick_params(width=1.5)
		axs[jj, ii].yaxis.set_tick_params(width=1.5)

		axs[jj, ii].spines["bottom"].set_linewidth(2)
		axs[jj, ii].spines["top"].set_linewidth(2)
		axs[jj, ii].spines["left"].set_linewidth(2)
		axs[jj, ii].spines["right"].set_linewidth(2)
	
		rectangle = patches.Rectangle((-15, 9), 36, 10, alpha=0.95, zorder=2,
					facecolor="cyan", linewidth=2)
		axs[jj, ii].add_patch(rectangle)

		axs[jj, ii].text(-14.5, 16, obs, weight='bold', fontsize=12, color='black')
		axs[jj, ii].text(-14.5, 13, main_header['SOURCE'], weight='bold', fontsize=12, color='black')
		

		axs[jj, ii].text(-2, 16, main_header['DATEOBS'][:-9], weight='bold', fontsize=12, color='black')
		axs[jj, ii].text(-2, 13, 'FWHM(a):{:.2f}"'.format(a_axis), weight='bold', fontsize=12, color='black')
		axs[jj, ii].text(-2, 10, 'FWHM(b):{:.2f}"'.format(b_axis), weight='bold', fontsize=12, color='black')

		ellipse = Ellipse(xy=(pos_source[0]-40, pos_source[1]-40), width=a_axis, height=b_axis, angle=theta,
			edgecolor='red', fc='None', lw=2.5)

		axs[jj, ii].add_patch(ellipse)

	except Exception as e:
		print(e)

	n_temp += 1


fig.delaxes(axs[2][4])

cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.9])
cb = fig.colorbar(im, cax=cbar_ax)

cb.ax.set_ylabel('Normalized Amplitude [lineal]')
