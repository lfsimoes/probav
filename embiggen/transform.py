# 
# Copyright (c) 2019 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the GPL License. See the LICENSE file for details.


import skimage



# [============================================================================]


def bicubic_upscaling(img):
	"""
	Compute a bicubic upscaling by a factor of 3.
	"""
	r = skimage.transform.rescale(img, scale=3, order=3, mode='edge',
	                              anti_aliasing=False, multichannel=False)
	# NOTE: Don't change these options. They're required by `baseline_upscale`.
	# http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rescale
	# http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp
	return r
	

