# 
# Copyright (c) 2019 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the GPL License. See the LICENSE file for details.


import os
from glob import glob
from zipfile import ZipFile

import numpy as np
import skimage

from ._tqdm import tqdm



# [============================================================================]


def highres_image(path, img_as_float=True):
	"""
	Load a scene's high resolution image and its corresponding status map.
	
	Returns a `(hr, sm)` tuple, where:
	* `hr`: matrix with the loaded high-resolution image (values as np.uint16 or
	        np.float64 depending on `img_as_float`),
	* `sm`: the image's corresponding "clear pixel?" boolean mask.
	
	Scenes' image files are described at:
	https://kelvins.esa.int/proba-v-super-resolution/data/
	"""
	path = path if path[-1] in {'/', '\\'} else (path + '/')
	hr = skimage.io.imread(path + 'HR.png', dtype=np.uint16)
	sm = skimage.io.imread(path + 'SM.png', dtype=np.bool)
	if img_as_float:
		hr = skimage.img_as_float64(hr)
	return (hr, sm)
	


def lowres_image_iterator(path, img_as_float=True):
	"""
	Iterator over all of a scene's low-resolution images (LR*.png) and their
	corresponding status maps (QM*.png).
	
	Returns at each iteration a `(l, c)` tuple, where:
	* `l`: matrix with the loaded low-resolution image (values as np.uint16 or
	       np.float64 depending on `img_as_float`),
	* `c`: the image's corresponding "clear pixel?" boolean mask.
	
	Scenes' image files are described at:
	https://kelvins.esa.int/proba-v-super-resolution/data/
	"""
	path = path if path[-1] in {'/', '\\'} else (path + '/')
	for f in glob(path + 'LR*.png'):
		q = f.replace('LR', 'QM')
		l = skimage.io.imread(f, dtype=np.uint16)
		c = skimage.io.imread(q, dtype=np.bool)
		if img_as_float:
			l = skimage.img_as_float64(l)
		yield (l, c)
	


# [============================================================================]


def check_img_as_float(img, validate=True):
	"""
	Ensure `img` is a matrix of values in floating point format in [0.0, 1.0].
	Returns `img` if it already obeys those requirements, otherwise converts it.
	"""
	if not issubclass(img.dtype.type, np.floating):
		img = skimage.img_as_float64(img)
	# https://scikit-image.org/docs/dev/api/skimage.html#img-as-float64
	
	if validate:
		# safeguard against unwanted conversions to values outside the
		# [0.0, 1.0] range (would happen if `img` had signed values).
		assert img.min() >= 0.0 and img.max() <= 1.0
	
	return img
	


# [============================================================================]


def all_scenes_paths(base_path):
	"""
	Generate a list of the paths to all scenes available under `base_path`.
	"""
	base_path = base_path if base_path[-1] in {'/', '\\'} else (base_path + '/')
	return [
		base_path + c + s
		for c in ['RED/', 'NIR/']
		for s in sorted(os.listdir(base_path + c))
		]
	


def scene_id(scene_path, incl_channel=False):
	"""
	Extract from a scene's path its unique identifier.
	
	Examples
	--------
	>>> scene_id('probav/train/RED/imgset0559/')
	'imgset0559'
	>>> scene_id('probav/train/RED/imgset0559', incl_channel=True)
	'RED/imgset0559'
	"""
	sep = os.path.normpath(scene_path).split(os.sep)
	if incl_channel:
		return '/'.join(sep[-2:])
	else:
		return sep[-1]
	


# [============================================================================]


def prepare_submission(images, scenes, subm_fname='submission.zip'):
	"""
	Prepare a set of images for submission.
	
	Given a list of `images` (as matrices of shape (384, 384)), and the paths
	to the `scenes` to which they correspond, write a zip file containing all
	images as .png files, named after their scene's identification
	(example: imgset1160.png).
	"""
	assert len(images) == 290, '%d images provided, 290 expected.' % len(images)
	assert len(images) == len(scenes), "Mismatch in number of images and scenes."
	assert subm_fname[-4:] == '.zip'
	
	# specific warnings we wish to ignore
	warns = [
		'tmp.png is a low contrast image',
		'Possible precision loss when converting from float64 to uint16']
	
	with np.warnings.catch_warnings():
		for w in warns:
			np.warnings.filterwarnings('ignore', w)
		
		print('Preparing submission. Writing to "%s".' % subm_fname)
		
		with ZipFile(subm_fname, mode='w') as zf:
			
			for img, scene in zip(tqdm(images), scenes):
				assert img.shape == (384, 384), \
					'Wrong dimensions in image for scene %s.' % scene
				
				skimage.io.imsave('tmp.png', img)
				zf.write('tmp.png', arcname=scene_id(scene) + '.png')
		
		os.remove('tmp.png')
	

