# 
# Copyright (c) 2019 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the GPL License. See the LICENSE file for details.


import numpy as np
import scipy
import skimage

from .io import lowres_image_iterator
from .transform import bicubic_upscaling



# [============================================================================]


def baseline_upscale(path):
	"""
	Reimplementation of the image enhancement operation performed by the
	baseline code (`generate_sample_submission.py`) provided in:
	https://kelvins.esa.int/proba-v-super-resolution/submission-rules/
	
		"takes all low resolution images that have the maximum amount of clear
		pixels, computes a bicubic upscaling by a factor of 3 and averages their
		pixel intensities."
	
	This function takes as argument the `path` to a single scene, and returns
	the matrix with the scene's enhanced image.
	"""
	clearance = {}
	for (l, c) in lowres_image_iterator(path, img_as_float=True):
		clearance.setdefault(c.sum(), []).append(l)
	
	# take all the images that have the same maximum clearance
	imgs = max(clearance.items(), key=lambda i: i[0])[1]
	
	sr = np.mean([
		bicubic_upscaling(i)
		for i in imgs
		], axis=0)
	
	return sr
	


def central_tendency(images, agg_with='median',
	                 only_clear=False, fill_obscured=False,
	                 img_as_float=True):
	"""
	Aggregate the given `images` through a statistical central tendency measure,
	chosen by setting `agg_with` to either 'mean', 'median' or 'mode'.
	
	Expects `images` to be a list of `(image, status map)` tuples.
	Should `images` be a string, it's interpreted as the path to a scene's
	files. The code will then aggregate that scene's low resolution images
	(LR*.png), while taking also into account their status maps (QM*.png).
	
	Will optionally aggregate only images' clear pixels (if `only_clear=True`)
	by using the information in images' corresponding status maps.
	
	In some scenes, some pixels are obscured in all of the low-resolution
	images. Aggregation with mean/median will return np.nan for those pixels,
	and aggregation with mode will return 0.0.
	If called with `fill_obscured=True` those pixels will be filled with the
	`agg_with` aggregate of the values at all those obscured pixels. Setting
	`fill_obscured` to one of 'mean', 'median' or 'mode' will indicate that is
	the measure that should be used to aggregate obscured pixels.
	"""
	agg_opts = {
		'mean'   : lambda i: np.nanmean(i, axis=0),
		'median' : lambda i: np.nanmedian(i, axis=0),
		'mode'   : lambda i: scipy.stats.mode(i, axis=0, nan_policy='omit').mode[0],
		}
	agg = agg_opts[agg_with]
	
	imgs = []
	obsc = []
	
	if isinstance(images, str):
		images = lowres_image_iterator(images, img_as_float or only_clear)
	elif only_clear:
		# Images were given by the caller, rather than loaded here.
		# Because `only_clear=True`, we generate copies of all lr images, so the
		# function will have no unintended side-effects on the caller's side.
		images = [(l.copy(), c) for (l,c) in images]
	
	for (l, c) in images:
		
		if only_clear:
			
			# keep track of the values at obscured pixels
			if fill_obscured != False:
				o = l.copy()
				o[c] = np.nan
				obsc.append(o)
			
			# replace values at obscured pixels with NaNs
			l[~c] = np.nan
		
		imgs.append(l)
	
	# aggregate the images
	with np.warnings.catch_warnings():   ## https://stackoverflow.com/a/29348184
		# suppress the warnings that originate when `only_clear=True`
		# but some pixels are never clear in any of the images
		np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
		np.warnings.filterwarnings('ignore', r'Mean of empty slice')
		
		agg_img = agg(imgs)
		
		if only_clear and fill_obscured != False:
			if isinstance(fill_obscured, str):
				agg = agg_opts[fill_obscured]
			some_clear = np.isnan(obsc).any(axis=0)
			obsc = agg(obsc)
			obsc[some_clear] = 0.0
			np.nan_to_num(agg_img, copy=False)
			agg_img += obsc
	
	return agg_img
	

