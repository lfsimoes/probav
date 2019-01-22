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
	

