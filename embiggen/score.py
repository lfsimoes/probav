# 
# Copyright (c) 2019 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the GPL License. See the LICENSE file for details.


import os

import numpy as np
import pandas as pd
import numba
import skimage

from .io import highres_image, check_img_as_float, all_scenes_paths, scene_id
from ._tqdm import tqdm



# [============================================================================]


# Baseline cPSNR values for the dataset's images. Used for normalizing scores.
# (provided by the competition's organizers)
baseline_cPSNR = pd.read_csv(
	os.path.dirname(os.path.abspath(__file__)) + '/norm.csv',
	names = ['scene', 'cPSNR'],
	index_col = 'scene',
	sep = ' ')
	


# [============================================================================]


def score_images(imgs, scenes_paths, *args):
	"""
	Measure the overall (mean) score across multiple super-resolved images.
	
	Takes as input a sequence of images (`imgs`), a sequence with the paths to
	the corresponding scenes (`scenes_paths`), and optionally a sequence of
	(hr, sm) tuples with the pre-loaded high-resolution images of those scenes.
	"""
	return np.mean([
#		score_image(*i)
		score_image_fast(*i)
		for i in zip(tqdm(imgs), scenes_paths, *args)
		])
	


def score_image(sr, scene_path, hr_sm=None):
	"""
	Calculate the individual score (cPSNR, clear Peak Signal to Noise Ratio) for
	`sr`, a super-resolved image from the scene at `scene_path`.
	
	Parameters
	----------
	sr : matrix of shape 384x384
		super-resolved image.
	scene_path : str
		path where the scene's corresponding high-resolution image can be found.
	hr_sm : tuple, optional
		the scene's high resolution image and its status map. Loaded if `None`.
	"""
	hr, sm = highres_image(scene_path) if hr_sm is None else hr_sm
	
	# "We assume that the pixel-intensities are represented
	# as real numbers ∈ [0,1] for any given image."
	sr = check_img_as_float(sr)
	hr = check_img_as_float(hr, validate=False)
	
	# "Let N(HR) be the baseline cPSNR of image HR as found in the file norm.csv."
	N = baseline_cPSNR.loc[scene_id(scene_path)][0]
	
	# "To compensate for pixel-shifts, the submitted images are
	# cropped by a 3 pixel border, resulting in a 378x378 format."
	sr_crop = sr[3 : -3, 3 : -3]
	
	crop_scores = []
	
	for (hr_crop, sm_crop) in hr_crops(hr, sm):
		# values at the cropped versions of each image that
		# fall in clear pixels of the cropped `hr` image
		_hr = hr_crop[sm_crop]
		_sr = sr_crop[sm_crop]
		
		# "we first compute the bias in brightness b"
		pixel_diff = _hr - _sr
		b = np.mean(pixel_diff)
		
		# "Next, we compute the corrected clear mean-square
		# error cMSE of SR w.r.t. HR_{u,v}"
		pixel_diff -= b
		cMSE = np.mean(pixel_diff * pixel_diff)
		
		# "which results in a clear Peak Signal to Noise Ratio of"
		cPSNR = -10. * np.log10(cMSE)
		
		# normalized cPSNR
		crop_scores.append(N / cPSNR)
#		crop_scores.append(cMSE)
	
	# "The individual score for image SR is"
	sr_score = min(crop_scores)
#	sr_score = N / (-10. * np.log10(min(crop_scores)))
	
	return sr_score
	


# [===================================]


def hr_crops(hr, sm):
	"""
	"We denote the cropped 378x378 images as follows: for all u,v ∈ {0,…,6},
	HR_{u,v} is the subimage of HR with its upper left corner at coordinates
	(u,v) and its lower right corner at (378+u, 378+v)."
	-- https://kelvins.esa.int/proba-v-super-resolution/scoring/
	"""
	for u in range(6):
		for v in range(6):
			yield hr[u : -6 + u, v : -6 + v], \
				  sm[u : -6 + u, v : -6 + v]
	


# [============================================================================]


def score_image_fast(sr, scene_path, hr_sm=None):
	"""
	Calculate the individual score (cPSNR, clear Peak Signal to Noise Ratio) for
	`sr`, a super-resolved image from the scene at `scene_path`.
	
	Parameters
	----------
	sr : matrix of shape 384x384
		super-resolved image.
	scene_path : str
		path where the scene's corresponding high-resolution image can be found.
	hr_sm : tuple, optional
		the scene's high resolution image and its status map. Loaded if `None`.
	"""
	hr, sm = highres_image(scene_path) if hr_sm is None else hr_sm
	
	# "We assume that the pixel-intensities are represented
	# as real numbers ∈ [0,1] for any given image."
	sr = check_img_as_float(sr)
	hr = check_img_as_float(hr, validate=False)
	
	# "Let N(HR) be the baseline cPSNR of image HR as found in the file norm.csv."
	N = baseline_cPSNR.loc[scene_id(scene_path)][0]
	
	return score_against_hr(sr, hr, sm, N)
	


@numba.jit('f8(f8[:,:], f8[:,:], b1[:,:], f8)', nopython=True, parallel=True)
def score_against_hr(sr, hr, sm, N):
	"""
	Numba-compiled version of the scoring function.
	"""	
	# "To compensate for pixel-shifts, the submitted images are
	# cropped by a 3 pixel border, resulting in a 378x378 format."
	sr_crop = sr[3 : -3, 3 : -3].ravel()
	
#	crop_scores = []
	cMSEs = np.zeros((6, 6), np.float64)
	
	for u in numba.prange(6):
		for v in numba.prange(6):
			
			# "We denote the cropped 378x378 images as follows: for all u,v ∈
			# {0,…,6}, HR_{u,v} is the subimage of HR with its upper left corner
			# at coordinates (u,v) and its lower right corner at (378+u, 378+v)"
			hr_crop = hr[u : -6 + u, v : -6 + v].ravel()
			sm_crop = sm[u : -6 + u, v : -6 + v].ravel()
			
			# values at the cropped versions of each image that
			# fall in clear pixels of the cropped `hr` image
			_sm_crop = np.where(sm_crop)
			_hr = hr_crop[_sm_crop]
			_sr = sr_crop[_sm_crop]
			
			# "we first compute the bias in brightness b"
			pixel_diff = _hr - _sr
			b = np.mean(pixel_diff)
			
			# "Next, we compute the corrected clear mean-square
			# error cMSE of SR w.r.t. HR_{u,v}"
			pixel_diff -= b
			cMSE = np.mean(pixel_diff * pixel_diff)
			
			# "which results in a clear Peak Signal to Noise Ratio of"
#			cPSNR = -10. * np.log10(cMSE)
			
			# normalized cPSNR
#			crop_scores.append(N / cPSNR)
			
			cMSEs[u, v] = cMSE
	
	# "The individual score for image SR is"
#	sr_score = min(crop_scores)
	sr_score = N / (-10. * np.log10(cMSEs.min()))
	
	return sr_score
	


# [============================================================================]


class scorer(object):
	
	def __init__(self, scene_paths, preload_hr=True):
		"""
		Wrapper to `score_image()` that simplifies the scoring of multiple
		super-resolved images.
		
		The scenes over which the scorer will operate should be given in
		`scene_paths`. This is either a sequence of paths to a subset of scenes
		or a string with a single path. In this case, it is interpreted as the
		base path to the full dataset, and `all_scenes_paths()` will be used to
		locate all the scenes it contains.
		
		Scene paths are stored in the object's `.paths` variable.
		When scoring, only the super-resolved images need to be provided.
		They are assumed to be in the same order as the scenes in `.paths`.
		
		If the object is instantiated with `preload_hr=True` (the default),
		all scene's high-resolution images and their status maps will be
		preloaded. When scoring they will be sent to `score_image()`, thus
		saving computation time in repeated scoring, at the expense of memory.
		"""
		if isinstance(scene_paths, str):
			self.paths = all_scenes_paths(scene_paths)
		else:
			self.paths = scene_paths
		
		self.hr_sm = [] if not preload_hr else [
			highres_image(scn_path, img_as_float=True)
			for scn_path in tqdm(self.paths, desc='Preloading hi-res images')]
		
		self.scores = []
		
	
	def __call__(self, sr_imgs, per_image=False, progbar=True, desc=''):
		"""
		Score all the given super-resolved images (`sr_imgs`), which correspond
		to the scenes at the matching positions of the object's `.paths`.
		
		Returns the overall score (mean normalized cPSNR).
		
		An additional value is returned if `per_image=True`: a list with each
		image's individual cPSNR score. In either case, this list remains
		available in the object's `.scores` variable until the next call.
		"""
		scenes_paths = tqdm(self.paths, desc=desc) if progbar else self.paths
		hr_sm = [] if self.hr_sm == [] else [self.hr_sm]
		
		self.scores = [
#			score_image(*i)
			score_image_fast(*i)
			for i in zip(sr_imgs, scenes_paths, *hr_sm)]
		
		assert len(self.scores) == len(self.paths)
		
		score = np.mean(self.scores)
		
		if per_image:
			return score, self.scores
		else:
			return score
		
	
