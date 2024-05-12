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
rc('font', family='serif', size='20')

ion()

# Initial arguments
# --------------------------------------------------------

# Band to analise [um]
band = "1100"

# Theoretical FWHM
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
obs_nums = ['094468', '094469', '094471', '098811', '099077', '099108', 
			'100280', '100286']

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


# C O A D D   M A P S
# ------------------------------------------------

flag_n = True

# Extract the data
n_temp = 0
headers, datas, weights, refs = {}, {}, {}, {}

for obs in obs_nums:

	for chn in chns:
		try:
			if not obs in flag_obs[chn]:
				# Filename
				filename = "muscat_commissioning_a"+band+"_"+obs+"_"+chn+".fits"	
				path = diry + '/' + filename
				# Read data
				main_header, header, data = read_map(path, ext) 

				# Default refs
				post_data, refs[n_temp] = get_source_refs(data[ext[0]], header[ext[0]], data[ext[2]], fwhm, thresh, norm=True)

				if refs[n_temp] != -1:
					# Create the useful dictionaries
					headers[n_temp] = header[ext[0]]
					datas[n_temp] = post_data
					weights[n_temp] = data[ext[1]]

					n_temp += 1

			else:
				print(obs, ' flagged!')

		except Exception as e:
			print(e)
			print(obs, ' flagged!')


sample_map, ref, deltas, weight_map = coadd_maps(headers, datas, weights, refs)

# Get stats from the coadd-map
centre_x, centre_y = ref[0], ref[1]
centre_sample_radius = 4*fwhm

point_source_coadd = sample_map[int(centre_y-centre_sample_radius):int(centre_y+centre_sample_radius)+1,
								int(centre_x-centre_sample_radius):int(centre_x+centre_sample_radius)+1]
xx, yy = np.meshgrid(np.linspace(0, 2*centre_sample_radius+1, 2*centre_sample_radius+1, endpoint=False),
									np.linspace(0, 2*centre_sample_radius+1, 2*centre_sample_radius+1, endpoint=False))

mask =~ np.isnan(point_source_coadd)

popt, pcov = fit_beam(xx, yy, point_source_coadd, 'ElGauss', mask=mask)


# S A V E   F I T S
# ------------------------------------------------

# Create Header
w = WCS(naxis=2)
w.wcs.crpix = [ref[0], ref[1]]
w.wcs.crval = [0, 0]
w.wcs.cdelt = np.array([deltas[0], deltas[1]])
pixscale_x, pixscale_y = deltas[0], deltas[1]
w.wcs.ctype = ['AZOFFSET', 'ELOFFSET']
w.wcs.cunit = ['arcsec  ', 'arcsec  ']
header_coadd = w.to_header()
header_coadd['UNIT'] = ''

# Create FITS file
comment = 'flagged-norm'
name_coadd = "Coadd_map-1100-Al-Az-pointing-MUSCAT-norm.fits"
hdu = fits.PrimaryHDU(sample_map, header=header_coadd)
hdu.writeto(name_coadd, overwrite=True)



# F I T   R E S U L T S
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



# L O G   S C A L E
# ----------------------------------------

fig_db, axs_db = subplots(1, 1)
subplots_adjust(left=0.010, right=0.885, top=0.935, bottom=0.06, hspace=0.0, wspace=0.0)


im_db = axs_db.imshow(10*np.log10((point_source_coadd-offset)/(A-offset)), cmap='jet', vmin=-50, extent=[-0.5-centre_sample_radius,centre_sample_radius+0.5,-0.5-centre_sample_radius,centre_sample_radius+0.5], origin='lower')

axs_db.contour(10*np.log10((point_source_coadd-offset)/(A-offset)), levels=[-50,-40,-30,-20,-10], linewidths=1.0, colors='k', extent=[-0.5-centre_sample_radius,centre_sample_radius+0.5,-0.5-centre_sample_radius,centre_sample_radius+0.5])

axs_db.set_ylim(-30, 30)
axs_db.set_xlim(-30, 30)

#axs_db.tick_params(axis="y",direction="in", length=10, pad=-22, colors='black')
#axs_db.tick_params(axis="x",direction="in", length=10, pad=-22, colors='black')

axs_db.xaxis.set_tick_params(width=1.5)
axs_db.yaxis.set_tick_params(width=1.5)

axs_db.spines["bottom"].set_linewidth(2)
axs_db.spines["top"].set_linewidth(2)
axs_db.spines["left"].set_linewidth(2)
axs_db.spines["right"].set_linewidth(2)

rectangle = patches.Rectangle((-25, -25), 21.5, 8, alpha=0.95, zorder=2,
			facecolor="green", linewidth=2)
axs_db.add_patch(rectangle)

axs_db.text(-24.5, -19.5, 'FWHM(a): {:.2f} "'.format(a_axis), weight='bold', fontsize=22, color='white')
axs_db.text(-24.5, -22, 'FWHM(b): {:.2f} "'.format(b_axis), weight='bold', fontsize=22, color='white')
axs_db.text(-24.5, -24.5, 'ecc: {:.2f}'.format(e), weight='bold', fontsize=22, color='white')

ellipse = Ellipse(xy=(pos_source[0]-40, pos_source[1]-40), width=a_axis, height=b_axis, angle=theta,
	edgecolor='white', fc='None', lw=2.5)

axs_db.add_patch(ellipse)

axs_db.set_ylabel('[arcsec]')
axs_db.set_xlabel('[arcsec]')

axs_db.xaxis.set_tick_params(labelsize=20)
axs_db.yaxis.set_tick_params(labelsize=20)

cb = fig_db.colorbar(im_db, orientation='vertical', fraction=0.046, pad=0.08)
cb.set_label('Normalized Amplitude [dB]', labelpad=-120)



# L I N   S C A L E
# ----------------------------------------

fig_lin, axs_lin = subplots(1, 1)
subplots_adjust(left=0.010, right=0.885, top=0.935, bottom=0.06, hspace=0.0, wspace=0.0)

im_lin = axs_lin.imshow(point_source_coadd, cmap='jet', vmin=-0.1, vmax=1, extent=[-0.5-centre_sample_radius,centre_sample_radius+0.5,-0.5-centre_sample_radius,centre_sample_radius+0.5], origin='lower')#, extent=[-30, 30, -30, 30])

axs_lin.contour(point_source_coadd, levels=7, linewidths=1.0, colors='k', extent=[-0.5-centre_sample_radius,centre_sample_radius+0.5,-0.5-centre_sample_radius,centre_sample_radius+0.5])

axs_lin.set_ylim(-30, 30)
axs_lin.set_xlim(-30, 30)

#axs_lin.tick_params(axis="y",direction="out", length=10, pad=-22, colors='black')
#axs_lin.tick_params(axis="x",direction="out", length=10, pad=-22, colors='black')

axs_lin.xaxis.set_tick_params(width=1.5)
axs_lin.yaxis.set_tick_params(width=1.5)

axs_lin.spines["bottom"].set_linewidth(2)
axs_lin.spines["top"].set_linewidth(2)
axs_lin.spines["left"].set_linewidth(2)
axs_lin.spines["right"].set_linewidth(2)

rectangle = patches.Rectangle((-25, -25), 21.5, 8, alpha=0.95, zorder=2,
			facecolor="green", linewidth=2)
axs_lin.add_patch(rectangle)

axs_lin.text(-24.5, -19.5, 'FWHM(a): {:.2f} "'.format(a_axis), weight='bold', fontsize=22, color='white')
axs_lin.text(-24.5, -22, 'FWHM(b): {:.2f} "'.format(b_axis), weight='bold', fontsize=22, color='white')
axs_lin.text(-24.5, -24.5, 'ecc: {:.2f}'.format(e), weight='bold', fontsize=22, color='white')

#ellipse = Ellipse(xy=(pos_source[0]-40, pos_source[1]-40), width=a_axis, height=b_axis, angle=theta,
#	edgecolor='red', fc='None', lw=2.5)

#axs_lin.add_patch(ellipse)

axs_lin.set_ylabel('[arcsec]')
axs_lin.set_xlabel('[arcsec]')

axs_lin.xaxis.set_tick_params(labelsize=20)
axs_lin.yaxis.set_tick_params(labelsize=20)

cb = fig_lin.colorbar(im_lin, orientation='vertical', fraction=0.046, pad=0.08)
cb.set_label('Normalized Amplitude', labelpad=-120)

