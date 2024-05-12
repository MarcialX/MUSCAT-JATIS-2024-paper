# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# PSF Functions
# psf_functions.py
# Functions for PSFF analysis
#
# Marcial Becerril, @ 14 March 2023
# Latest Revision: 14 Mar 2023, 22:53 GMT-6
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #


from scipy import optimize
from scipy.optimize import curve_fit

import numpy as np

from matplotlib.pyplot import *
ion()



def B3G(x, *params):
	"""
		3D gaussian profile as NIKA2 Paper
		---------
		Parameters
			Input:
				x: array
					x data
				params: list
					Data grouped in 2 items for every guassian profile
					plus a single parameters for the baseline level
	"""
	y = np.zeros_like(x)
	for i in range(0, len(params)-1, 2):
		#x0 = params[i+2]
		x0 = 0.0	# All pattern centered
		amp = params[i]
		sig = params[i+1]
		y += amp * np.exp( -((x - x0)/sig)**2)

	# Last item accounts for baseline
	B0 = params[-1]
	y += B0

	return y


def twoD_Gaussian(pos, amplitude, xo, yo, sigma, offset):
	"""
		2D Gaussian function.
		---------
		Parameters
			Input:
				pos:     	[list] Points of the 2D map
				amplitude:  [float] Amplitude
				xo, yo:     [float] Gaussian profile position
				sigma:      [float] Dispersion profile
				offset:     [float] Offset profile
			Ouput:
				g:			[list] Gaussian profile unwraped in one dimension
	"""
	x = pos[0]
	y = pos[1]
	xo = float(xo)
	yo = float(yo)
	g = offset + amplitude*np.exp(-(((x - xo)**2/2./sigma**2) + ((y - yo)**2/2./sigma**2)))
	return g.ravel()


def twoD_ElGaussian(pos, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
	"""
		2D Elliptical Gaussian function.
		---------
		Parameters
			Input:
				pos:             	[list] Points of the 2D map
				amplitude:          [float] Amplitude
				xo, yo:             [float] Gaussian profile position
				sigma_x, sigma_y:   [float] X-Y Dispersion profile
				theta:              [float] Major axis inclination
				offset:             [float] Offset profile
			Ouput:
				g:			[list] Gaussian profile unwraped in one dimension
	"""
	x = pos[0]
	y = pos[1]
	xo = float(xo)
	yo = float(yo)
	a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
	b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
	c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
	g = offset+amplitude*np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
	return g.ravel()


# R A D I A L   P R O F I L E   F U N C T I O N S
# ------------------------------------------------
def get_slice(data, xc, yc, radius, theta, full=True, npoints=100):
	"""
		Get a slice from a map
		---------
		Parameters
			Input:
				data: array
					Data map from where the slice will be extracted
				xc, yc: float
					Center of the slice
				radius: float
					Radius of the slice cut. Two times the radius is the size of the slice
				theta: float
					Angle of the cut
				full: bool
					Half slice (starting from xc, yc) or full slice
				npoints: int
					Number of points of the slice
			Output: 
				r_filt: array
					Radius of slice
				data_slice: array
					Slice data
	"""
	# Dimensions of data
	xdim, ydim = data.shape
	# Radius points
	delta_rad = np.linspace(0, radius, npoints)
	r = np.copy(delta_rad)
	# Get the points of the slice
	x = delta_rad*np.cos(theta*np.pi/180)
	y = delta_rad*np.sin(theta*np.pi/180)

	# Do we want a 2-quadrants slice?
	if full: 
		r = np.concatenate((-r, r))
		sort_r = np.argsort(r)
		r = r[sort_r]
		x = np.concatenate((-x, x))
		y = np.concatenate((-y, y))
		x = x[sort_r]
		y = y[sort_r]

	# Convert to pixels
	x = (x+xc).astype(int)
	y = (y+yc).astype(int)

	# Filter the duplicate points
	x_filt, y_filt = [], []
	r_filt = []
	for p in range(len(x)):
		if not (x[p] in x_filt and y[p] in y_filt):
			x_filt.append(x[p])
			y_filt.append(y[p])
			r_filt.append(r[p])

	x_filt = np.array(x_filt)
	y_filt = np.array(y_filt)
	r_filt = np.array(r_filt)

	#imshow(data)
	#plot(x,y,'r.')

	data_slice = data[x_filt, y_filt]

	return r_filt, data_slice


def get_radial_profile(pos, point_source_data, pixscale=1, radius=20, full=True, norm=True, mix='mean', npoints=100, rad_points=30):
	"""
		Get the combined radial profile of a source
		---------
		Parameters
			Input:
				pos: list
					Centre position of profile
				point_source_data: array
					Map from where thr profile is extracted
				pixscale: float
					Pixel scale in arcsec
				radius: float
					Profile radius
				full: bool
					Half slice (starting from xc, yc) or full slice
				norm: bool
					Normalise the PSF
				mix: string
					'mean': average the slices
					'med': get the median of slices
				npoints: int
					Number of points of the slice
				rad_points: int
					Number of points of the radius
			Output: 
				radius_com: array
					Radius of radial slice
				profile: array
					Data profile
	"""

	# Initial setup
	radius_pix = int(radius/3600/pixscale)

	# 1. Extract a data sample centered in the source
	x = pos[0]
	y = pos[1]

	if full:
		up_limit = 180
		# Combined radius
		radius_com = np.linspace(-radius_pix, radius_pix, rad_points)
	else:
		up_limit = 360
		# Combined radius
		radius_com = np.linspace(0, radius_pix, rad_points)

	slices_theta = [[] for i in range(rad_points)]
	d_rad_com = radius_com[1]-radius_com[0]
	#print(d_rad_com)

	# Get a slice
	for t in np.linspace(0, up_limit, npoints):
		rad, slice_theta = get_slice(point_source_data, x, y, radius_pix, t, full=full)

		n_idx = (rad/d_rad_com).astype(int)
		if full:
			n_idx += np.abs(n_idx[0])

		for i in range(len(rad)):
			slices_theta[n_idx[i]].append(slice_theta[i])

	profile = np.linspace(0, radius, rad_points)
	for i in range(rad_points):
		if mix == 'mean':
			profile[i] = np.nanmean(slices_theta[i])
		elif mix == 'med':
			profile[i] = np.nanmedian(slices_theta[i])

	if norm:
		# Get limits
		max_profile = np.nanmax(profile)
		min_profile = np.nanmin(profile)

		k_factor = max_profile-min_profile
		profile = (profile-min_profile)/k_factor

	return radius_com, profile


def guess_beam_params(data, fit_func, sig=2.5):
	"""
	    Guess the basic parameters from the raw data
	        kid: (string) detector the guess parameters
	"""
	# Max ID
	max_idx = np.where(data==np.nanmax(data))
	# Amplitude
	amp = np.nanmax(data)
	# Offset position
	x0 = max_idx[1][0]
	y0 = max_idx[0][0]
	# Offset level
	offset = np.nanmean(data)

	if fit_func == 'Gauss':
		# Sigma
		# 2 pixels by default
		sigma = sig
		return [amp, x0, y0, sigma, offset]

	elif fit_func == 'ElGauss':
		# Sigma X-Y
		sigma_x = sig
		sigma_y = sig
		# Theta
		theta = 0.
		return [amp, x0, y0, sigma_x, sigma_y, theta, offset]


def fit_beam(x, y, data, func, mask=None):

	if not mask is None:
		data_fit = data[mask]
		x_fit = x[mask]
		y_fit = y[mask]

		print('Estoy enmascarando!')

	else:
		data_fit = data
		x_fit = x
		y_fit = y

	# Initial guess
	initial_guess = guess_beam_params(data, func)

	# Fitting
	if func == 'Gauss':
		popt, pcov = optimize.curve_fit(twoD_Gaussian, (x_fit, y_fit), data_fit.ravel(), p0=initial_guess)
	elif func == 'ElGauss':
		popt, pcov = optimize.curve_fit(twoD_ElGaussian, (x_fit, y_fit), data_fit.ravel(), p0=initial_guess)

	return popt, pcov
