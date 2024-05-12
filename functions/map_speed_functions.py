# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# Mapping speed functions
# map_speed_functions.py
#
# Marcial Becerril, @ 25 May 2023
# Latest Revision: 25 May 2023, 15:04 GMT-6
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

# Import PSF functions
import sys
sys.path.append('/home/marcial/PSF-extraction')

from psf_functions import *


def read_map(path, extensions, dims=2, block_ext=False):
	"""
		Read FITS maps
		---------
		Parameters
			Input:
				path:		[string] Filename data FITS
				extensions: [list of strings] Name extensions 
			Ouput:
				main_header:	[header obj] Main file header
				header_ext:		[list of headers] Headers of each extension
				data_ext:		[list of matrix] Data for each extension
	"""
	# Extract FITS data
	with fits.open(path) as hdu:
		# Get Genaral Header
		main_header = hdu[0].header

		if block_ext:
			data = hdu[0].data
			return main_header, data

		# Extract extensions
		header_ext = {}
		data_ext = {}
		for ext in extensions:
			# Key name
			name = ext
			# H E A D E R
			header_ext[name] = {}
			header_ext[name] = hdu[ext].header
			# D A T A
			# Ignore empty dimensions
			data = hdu[ext].data
			for i in range(data.ndim-dims):
				data = data[0,:,:]
			data_ext[name] = {}
			data_ext[name] = data

	return main_header, header_ext, data_ext


def map_std(data, fwhm_a, fwhm_b, mask, method="median"):
	"""
		Get a std of a beam area along the whole map
	"""
	# Calculate beam area [assuming a Guassin beam]
	beam = np.pi*(fwhm_a*fwhm_b)/(4*np.log(2))

	# Get a, b dimensions of a rectangular beam area
	a = int(np.round(np.sqrt(beam)))
	b = int(np.round(beam/a))
	# Used area
	print("Beam[pix]: ", beam)
	print("Used[pix]: ", a*b)

	# Get dimensions
	x, y = data.shape
	std_map = np.nan*np.zeros((x-a, y-b))
	for i in range(x-a):
		for j in range(y-b):
			sample_area = data[i:i+a, j:j+b]
			sample_mask = mask[i:i+a, j:j+b]
			if not (np.isnan(np.sum(sample_area)) or False in sample_mask):
				std_map[i][j] = np.std(sample_area)

	if method == "median":
		median_val = np.nanmedian(std_map)	

		idx_sort = np.argsort(std_map.flatten())
		sub_idx = np.where(median_val>=np.sort(std_map.flatten()))[0][-1]
		best_pos = idx_sort[sub_idx]
		print(best_pos)
		print(median_val)

	elif method == "mean":
		mean_val = np.nanmean(std_map)	

		idx_sort = np.argsort(std_map.flatten())
		sub_idx = np.where(mean_val>=np.sort(std_map.flatten()))[0][-1]
		best_pos = idx_sort[sub_idx]
		print(best_pos)
		print(mean_val)

	elif method == "min":
		best_pos = np.nanargmin(std_map)
		print(np.nanmin(std_map))

	b_best = int(best_pos/(y-b))
	a_best = best_pos%(y-b)

	return std_map, a_best, b_best, a, b


def nefd_from_map(map_data, coverage_data, mask):
	"""
		Getting NEFD from the map
	"""
	# Sigma from the coverage map
	sigma = np.std(map_data[mask])
	# Get average integrated time for the region
	tavg = np.mean(coverage_data[mask])

	# Calculate the NEFD [as NIKA-2/Andreas/Emmaly]
	NEFD = sigma*np.sqrt(tavg)

	return NEFD, sigma, tavg


def log_sigma(t, NEFD):
	"""
		Getting logaritmic NEFD
	"""
	# Calculate the NEFD [as NIKA-2/Andreas/Emmaly]
	sigma = NEFD - 0.5*t

	return sigma

# No log-scale
def lin_sigma(t, NEFD):
	"""
		Getting logaritmic NEFD
	"""
	# Calculate the NEFD [as NIKA-2/Andreas/Emmaly]
	sigma = NEFD/np.sqrt(t)

	return sigma


def log_sigma_index(t, NEFD, index):
	"""
		Getting logaritmic NEFD with an index
	"""
	# Calculate the NEFD [as NIKA-2/Andreas/Emmaly]
	sigma = np.log10(NEFD) - index*np.log10(t)

	return sigma


def coadd_maps(headers, maps, weights, ref_ext=None, coverage=False):
	"""
		Co-add maps from different observations.
		---------
		Parameters
			Input:
				map_list:	[list of string] Filename data FITS
					The first map is taken as the reference map
			Ouput:

	"""

	# Map list
	obs_list = list(headers.keys())

	x1, y1 = 0, 0
	x2, y2 = 0, 0

	# Get the biggest dimensions
	for i, obs in enumerate(obs_list):
		
		if ref_ext is None:
			ref_ext_x = 0
			ref_ext_y = 0
		else:
			ref_ext_x = ref_ext[obs][0]
			ref_ext_y = ref_ext[obs][1]

		wcsobj = WCS(headers[obs]).sub(2)
		pixscale_x = (wcsobj.proj_plane_pixel_scales()[0]).to(u.arcsec).value
		pixscale_y = (wcsobj.proj_plane_pixel_scales()[1]).to(u.arcsec).value

		#print(pixscale_x)
		#print(pixscale_y)

		# Offset due to CRVAL
		headers[obs]['CRPIX1'] = (headers[obs]['NAXIS1']+1)/2 # -headers[obs]['CRVAL1']/headers[obs]['CDELT1']
		headers[obs]['CRPIX2'] = (headers[obs]['NAXIS2']+1)/2 # -headers[obs]['CRVAL2']/headers[obs]['CDELT2']

		headers[obs]['CRVAL1'] = 0
		headers[obs]['CRVAL2'] = 0

		print(headers[obs]['CRPIX1'], headers[obs]['CRVAL1'], pixscale_x)
		print(headers[obs]['CRPIX2'], headers[obs]['CRVAL2'], pixscale_y)

		# X positions
		size_x = headers[obs]['NAXIS1']
		delta_x = pixscale_x*np.sign(headers[obs]['CDELT1']) # headers[obs]['CDELT1']/pixscale.value

		#min_x = -1*((delta_x*headers[obs]['CRPIX1'])+ref_ext_x)
		min_x = np.sign(delta_x)*(ref_ext_x-headers[obs]['CRPIX1'])
		max_x = size_x*np.sign(delta_x) + min_x

		# Get the minimum
		if i == 0:
			x1 = min_x
			x2 = max_x
		else:
			sign_x = np.sign(delta_x)
			if sign_x > 0:
				x1 = np.min([min_x, x1])
				x2 = np.max([max_x, x2])
			else:
				x1 = np.max([min_x, x1])
				x2 = np.min([max_x, x2])

		# Y positions
		size_y = headers[obs]['NAXIS2']
		delta_y = pixscale_y*np.sign(headers[obs]['CDELT2']) # headers[obs]['CDELT2']#/pixscale.value

		#print(delta_y)

		#min_y = -1*((delta_y*headers[obs]['CRPIX2'])+ref_ext_y)
		min_y = np.sign(delta_y)*(ref_ext_y-headers[obs]['CRPIX2'])
		max_y = size_y*np.sign(delta_y) + min_y

		# Get the minimum
		if i == 0:
			y1 = min_y
			y2 = max_y
		else:
			y1 = np.min([min_y, y1])
			y2 = np.max([max_y, y2])

	#print(x1, x2, y1, y2)

	# Create the canvas
	ref_x = x1 
	ref_y = y1

	coadded_map = np.zeros((int(np.abs(y2-y1)), int(np.abs(x2-x1))))
	weights_map = np.zeros_like(coadded_map)
	if coverage != False:
		coverage_map = np.zeros_like(coadded_map)

	#print(y2-y1, x2-x1)
	#print(coadded_map.shape)

	# Now fill the map!
	# Reference image
	ref_wcsobj = WCS(headers[obs_list[0]]).sub(2)
	ref_pixscale_x = (ref_wcsobj.proj_plane_pixel_scales()[0]).to(u.arcsec).value
	ref_pixscale_y = (ref_wcsobj.proj_plane_pixel_scales()[1]).to(u.arcsec).value

	ref_delta_x = ref_pixscale_x*np.sign(headers[obs_list[0]]['CDELT1']) # headers[obs_list[0]]['CDELT1']/pixscale.value
	ref_delta_y = ref_pixscale_y*np.sign(headers[obs_list[0]]['CDELT2']) # headers[obs_list[0]]['CDELT2']/pixscale.value

	for i, obs in enumerate(obs_list):
		
		wcsobj = WCS(headers[obs]).sub(2)
		pixscale_x = (wcsobj.proj_plane_pixel_scales()[0]).to(u.arcsec).value
		pixscale_y = (wcsobj.proj_plane_pixel_scales()[1]).to(u.arcsec).value

		# Deltas
		delta_x = pixscale_x*np.sign(headers[obs]['CDELT1']) # headers[obs]['CDELT1']/pixscale.value
		delta_y = pixscale_y*np.sign(headers[obs]['CDELT2']) # headers[obs]['CDELT2']/pixscale.value
		# Sizes
		size_x = headers[obs]['NAXIS1']
		size_y = headers[obs]['NAXIS2']
		# References
		if ref_ext is None:
			ref_ext_x = 0
			ref_ext_y = 0
		else:
			ref_ext_x = ref_ext[obs][0]
			ref_ext_y = ref_ext[obs][1]

		ref_x_temp = np.sign(delta_x)*(ref_ext_x-headers[obs]['CRPIX1'])
		ref_y_temp = np.sign(delta_y)*(ref_ext_y-headers[obs]['CRPIX2'])
	
		#print(ref_x, ref_x_temp)

		x = np.arange(size_x)
		x_coadd = x*(ref_delta_x/delta_x)+delta_x*(ref_x_temp-ref_x)
		y = np.arange(size_y)
		y_coadd = y*(ref_delta_y/delta_y)+delta_y*(ref_y_temp-ref_y)
		#print(x_coadd, y_coadd)

		x_coadd = x_coadd.astype(int)
		y_coadd = y_coadd.astype(int)

		xx, yy = np.meshgrid(x_coadd, y_coadd)

		#print(xx, yy)

		coadded_map[yy, xx] += maps[obs]*weights[obs]
		weights_map[yy, xx] += weights[obs]
		if coverage:
			coverage_map[yy, xx] += coverage[obs]

	coadded_map = coadded_map/weights_map

	if coverage:
		return coadded_map, coverage_map, (-np.sign(ref_delta_x)*ref_x, -np.sign(ref_delta_y)*ref_y), (ref_delta_x, ref_delta_y), weights_map
	else:
		return coadded_map, (-np.sign(ref_delta_x)*ref_x, -np.sign(ref_delta_y)*ref_y), (ref_delta_x, ref_delta_y), weights_map	


def _add_array(A, n, d):
	"""
		Add n rows to the matrix A
	"""
	y_length, x_length = np.shape(A)
	if d in ['u', 'd']:
		sample = np.zeros((1,x_length))
	elif d in ['l', 'r']:
		sample = np.zeros((y_length,1))
	else:
		print("Direction not valid")
		return

	for i in range(n):
		if d == 'd':
			A = np.r_[A, sample]
		elif d == 'u':
			A = np.r_[sample, A]
		elif d == 'l':
			A = np.c_[sample, A]
		elif d == 'r':
			A = np.c_[A, sample]

	return A


# Calculate the expected amplitude 
def expected_amplitude(image, sx, sy, h=None, k=None, neg_flag=False, aperture=6.5, ring=False, ring_rad=7.5, thickness=17.5):

	sx = sx/np.sqrt(8*np.log(2))
	sy = sy/np.sqrt(8*np.log(2))

	# Get the volume
	mask_source = np.zeros_like(image, dtype=bool)
	
	if h is None:
		h = int(len(mask_source)/2)

	if k is None:
		k = int(len(mask_source[0])/2)

	image_copy = np.copy(image)

	for l in range(len(mask_source)):
		for m in range(len(mask_source[0])):
			if ring:
				if ((l-h)**2 + (m-k)**2) > (ring_rad)**2 and ((l-h)**2 + (m-k)**2) < (ring_rad+thickness)**2:
					if neg_flag:
						if image[l][m] < 0:
							image[l][m] = 0
					#plot(l, m, 'bo')
					mask_source[l][m] = True
					image_copy[l][m] = 0.0					

			if ((l-h)**2 + (m-k)**2) < (aperture)**2 and image_copy[l][m]>30:
				if neg_flag:
					if image[l][m] < 0:
						image[l][m] = 0
				#plot(l, m, 'r*')
				mask_source[l][m] = True
			else:
				image_copy[l][m] = 0.0

	"""
	figure()
	imshow(image_copy, origin='lower')
	"""

	# Integrate the data inside the aperture
	V = np.sum(image[mask_source])
	#imshow(image_copy, origin='lower')

	# Expected amplitude
	A = V/(2*np.pi*sx*sy)

	return A


def get_source_refs(data, header, filt_data, fwhm, thresh, norm=True, frame=20):

	# Fit the Gaussian and get the offsets
	# ---------------------------------------------
	# 1. Remove the background
	# Get background noise through 'sigma-clipping'
	mean, median, std = sigma_clipped_stats(data[frame:-frame, frame:-frame], sigma=3.0)

	# 2. Look for sources...
	# Get pixel size
	wcsobj = WCS(header).sub(2)
	pixscale = (wcsobj.proj_plane_pixel_scales()[0]).to(u.arcsec)

	daofind = DAOStarFinder(fwhm=fwhm/pixscale.value, threshold=thresh*std)
	# Ignore the borders
	filt_x, filt_y = filt_data.shape
	z_filt = np.copy(filt_data)
	for i in range(filt_x):
		for j in range(filt_y):
			if (i < frame or i > filt_x-frame) or (j < frame or j > filt_y-frame):
				z_filt[i,j] = 0.0

			if np.isnan(z_filt[i,j]):
				z_filt[i,j] = 0.0

	sources = daofind(z_filt)

	print(sources)

	"""
	figure()
	imshow(filt_data)
	#figure()
	#imshow(z_filt, vmin=0, vmax=150)
	print(std, thresh)
	print('-->', pixscale.value)
	print('-->', sources)
	"""

	flag_source = False

	# Get the brightest source
	m = 1
	s_sort = sources.argsort('flux')
	for s in range(len(sources)):

		source_idx = s_sort[-m]

		#plot(sources['xcentroid'][source_idx], sources['ycentroid'][source_idx], 'r*')

		if filt_data[int(sources['ycentroid'][source_idx])][int(sources['xcentroid'][source_idx])] >= 100:
			# No-Minkasi: if (sources['xcentroid'][source_idx] > 200) and (sources['xcentroid'][source_idx] < 450) and (sources['ycentroid'][source_idx] > 200) and (sources['ycentroid'][source_idx] < 450):
			if (sources['xcentroid'][source_idx] > 100) and (sources['xcentroid'][source_idx] < 350) and (sources['ycentroid'][source_idx] > 100) and (sources['ycentroid'][source_idx] < 350):
				flag_source = True
				break
		m += 1

	if flag_source:
		pos_source = (sources['xcentroid'][source_idx], sources['ycentroid'][source_idx])
	else:
		xs, ys = filt_data.shape

		mx_pt = np.nanargmax(filt_data)
		i_mx = mx_pt%ys
		j_mx = int(mx_pt/ys)

		pos_source = (i_mx, j_mx)
		flag_source = True

	"""
	figure()
	imshow(z_filt)
	plot(pos_source[0], pos_source[1], 'r*')
	"""

	if flag_source: 
		
		print(pos_source)

		# 3. Fit beam
		# Get fitted position
		radius_pix = int(2.5*fwhm/pixscale.value)

		point_source_data = data[int(pos_source[1]-radius_pix):int(pos_source[1]+radius_pix)+1, int(pos_source[0]-radius_pix):int(pos_source[0]+radius_pix)+1]

		point_source_filt = filt_data[int(pos_source[1]-radius_pix):int(pos_source[1]+radius_pix)+1, int(pos_source[0]-radius_pix):int(pos_source[0]+radius_pix)+1]
		xx, yy = np.meshgrid(np.linspace(0, 2*radius_pix+1, 2*radius_pix+1, endpoint=False),
								np.linspace(0, 2*radius_pix+1, 2*radius_pix+1, endpoint=False))

		
		"""
		figure()
		imshow(point_source_data, origin='lower', cmap='rainbow')
		"""

		mask =~ np.isnan(point_source_data)

		popt, pcov = fit_beam(xx, yy, point_source_data, 'ElGauss', mask=mask)
		print(popt)


		pos_source_sample = (popt[1]+int(pos_source[0]-radius_pix), popt[2]+int(pos_source[1]-radius_pix))

		#figure()
		#imshow(data, origin='lower')
		#plot(pos_source_sample[0], pos_source_sample[1], 'k*')
		#print('-->', pos_source_sample)

		y_len, x_len = np.shape(data)

		#plot(x_len/2, y_len/2, 'r*')

		# Normalise data
		if norm:
			max_pixel = np.max(point_source_data)
			data = data/max_pixel

		return data, ((x_len/2 - pos_source_sample[0]), (y_len/2 - pos_source_sample[1]))

	else:

		return data, -1
