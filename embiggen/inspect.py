# 
# Copyright (c) 2019 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the GPL License. See the LICENSE file for details.


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

from collections import Counter, defaultdict
from scipy.stats import percentileofscore

from .aggregate import central_tendency
from .score import score_image_fast, hr_crops
from .io import highres_image, scene_id



# [============================================================================]


def describe(data, **kwargs):
	"""
	Generates descriptive statistics that summarize the central tendency,
	dispersion and shape of a dataset's distribution, excluding NaN values.
	
	Basic wrapper to pandas' `describe()`, to which extra arguments are
	redirected. See:
	https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html
	"""
	df = pd.DataFrame(data).describe(**kwargs)
	df.columns = ['']
	return df
	


def cdf_plot(x, **kwargs):
	"""
	Empirical Cumulative Distribution Function.
	https://en.wikipedia.org/wiki/Empirical_distribution_function
	"""
	n = np.arange(1, len(x) + 1) / np.float(len(x))
	plt.step(np.sort(x), n, **kwargs)
	plt.title('Empirical Cumulative Distribution Function')
	plt.xlabel('X')
	plt.ylabel('Cumulative probability')
	


def score_summary(*score_sets, labels=None, **kwargs):
	"""
	Generate a table of summary statistics and an empirical cumulative
	distribution plot for one or more result sets.
	
	Examples
	--------
	# Statistics for a single set of results, no labels:
	>>> (mean_score, img_scores) = score(sr_imgs, per_image=True)
	>>> score_summary(img_scores)
	
	# Comparison of different named sets of results:
	>>> score_summary(test1_scores, test2_scores, labels=['test 1', 'test 2'])
	
	Labels, if given, must be sent though the `labels` keyword argument.
	"""
	labeled = labels is not None
	if not labeled:
		labels = [''] * len(score_sets)
	if isinstance(labels, str):
		labels = [labels]
	
	assert not labeled or len(score_sets) == len(labels)
	
	with sns.axes_style('darkgrid'):
		with sns.plotting_context('notebook', font_scale=1.025):
			
			for sc, lb in zip(score_sets, labels):
				cdf_plot(sc, label=lb, **kwargs)
			if labeled:
				plt.legend()
			plt.xlabel('cPSNR')
	
	df = pd.DataFrame(list(zip(*score_sets)), columns=labels).describe()
	df.index.name = 'cPSNR'
	return df
	


# [============================================================================]


def create_panel(ncols=2, nrows=1):
	"""
	Initialize a panel with `ncols` columns and `nrows` rows.
	Configures the axes for image display: no ticks, no frame and equal aspect.
	
	Usage example:
	>>> fig, axs = create_panel(2, 2)
	>>> axs.flat[3].imshow(image)
	"""
	ax_cfg = dict(xticks=[], yticks=[], aspect='equal', frame_on=False)
	# additional options at:
	# https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.add_subplot
	
	fig, axs = plt.subplots(nrows, ncols,
	                        figsize=(5 * ncols, 5 * nrows),
	                        subplot_kw=ax_cfg)
	fig.tight_layout()
	
	return fig, axs
	


def compare_images(a, b, d=None):
	"""
	Compare side-by-side the images `a` and `b`.
	Shows on a third panel the squared difference between both, while
	compensating for the bias in brightness.
	"""
	fig, axs = create_panel(ncols=3, nrows=1)
	
	if d is None:
		d = a - b
		d -= np.mean(d)
		d *= d
	
	for ax, img, args in zip(axs.flat,
	                         [a, b, d],
	                         [{}, {}, dict(cmap=plt.cm.gray_r, vmin=0)]):
		
		ax.imshow(img, **args)
	
	return fig, axs
	


def compare_to_hr(sr, scene, only_clear=True):
	"""
	Compare side-by-side a super-resolved image `sr` of a given `scene` and its
	ground-truth (`hr`, the scene's high-resolution image).
	
	Shows on a third panel the squared difference between both, while
	compensating for the bias in brightness (`b`). Conceals by default
	(`only_clear=True`) the pixels that are obscured in the `hr` image and don't
	therefore interfere in the determination of `sr`'s cPSNR.
	
	Applies the same registration process that is employed by the competition's
	scoring function, which means the displayed images have a total of 6 pixels
	cropped along the edges in each dimension.
	
	The scene's cPSNR value shown on the left panel relates to the (rounded)
	`mean` value on the right panel (if `only_clear=True`) as:
	>>> baseline_cPSNR.loc[scene_id(scene)] / (-10. * np.log10(mean))
	"""
	(hr, sm) = highres_image(scene)
	
	sr_score = score_image_fast(sr, scene, (hr, sm))
	
	# image registration
	sr = sr[3 : -3, 3 : -3]
	min_cmse = None
	for (_hr, _sm) in hr_crops(hr, sm):
		d = _hr - sr
		if only_clear:
			d[~_sm] = np.nan
		d -= np.nanmean(d)
		d *= d
		m = np.nanmean(d)
		if min_cmse is None or m < min_cmse[0]:
			min_cmse = (m, d, _hr)
	(m, d, hr) = min_cmse
	
	fig, axs = compare_images(sr, hr, d)
	
	axs[0].set_title('super-resolved image, cPSNR: %.4f' % sr_score)
	axs[1].set_title('high-resolution image (ground-truth)')
	axs[2].set_title('(hr - sr - b)^2\n' + \
		'mean: %.2e, std: %.2e, max: %.2e' % (m, np.nanstd(d), np.nanmax(d)),
		fontdict=dict(verticalalignment='center'))
	
	# display the scene's id to the left of the image
	axs[0].text(-.01, 0.5, scene_id(scene, incl_channel=True),
		horizontalalignment='right',
		verticalalignment='center',
		rotation='vertical',
		transform=axs[0].transAxes)
	
	return fig, axs
	


# [============================================================================]


def compare_aggregates(img_path,
	                   type_opts=('mean', 'median', 'mode'),
	                   clear_opts=(False, True),
	                   fill_opts=(False, True)):
	"""
	Plot side-by-side the mean/median/mode aggregations of all images in a given
	scene. Also compares aggregation with and without occluded pixels.
	"""
	figs = []
	
	for only_clear in clear_opts:
		for fill_obscured in (fill_opts if only_clear else [False]):
			
			fig = plt.figure(figsize=(5 * len(type_opts), 6))
			figs.append(fig)
			
			for i, agg_type in enumerate(type_opts):
				
				args = (agg_type, only_clear, fill_obscured)
				img = central_tendency(img_path, *args)
				
				ax = fig.add_subplot(1, len(type_opts), i + 1)
				ax.imshow(img)
				ax.set_title(agg_type)
				ax.axis('off')
			
			t = 'only_clear=%s' % str(only_clear)
			if only_clear:
				t += ', fill_obscured=%s' % str(fill_obscured)
			fig.suptitle(t)
			fig.tight_layout()
	
	return figs
	

