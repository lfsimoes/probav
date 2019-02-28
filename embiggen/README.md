
# Contents #


[`io.py`][io.py]:

Loading the dataset's images, generating scene listings and identifiers, preparing submissions.

[`transform.py`][transform.py]:

Transformations over single images.

[`aggregate.py`][aggregate.py]:

Processing groups of images from the same scene.

[`score.py`][score.py]:

Implementation of the competition's scoring function, according to the specifications given at:
* https://kelvins.esa.int/proba-v-super-resolution/scoring/.

[`inspect.py`][inspect.py]:

Code for statistical or visual analysis of single images and datasets.

&nbsp;

[`norm.csv`][norm.csv]:

Provided by the competition's organizers.
Contains baseline cPSNR values for the dataset's images. Used for normalizing scores.
Downloaded from:
* https://kelvins.esa.int/proba-v-super-resolution/data/
* https://kelvins.esa.int/media/competitions/proba-v-super-resolution/probav_data.zip


[`_tqdm.py`][_tqdm.py]:

A distribution of the [original version of the `tqdm` module][tqdm], with minor modifications.
Simple and effective progress bar that just works (the current official module breaks from time to time on Jupyter notebooks).



[io.py]: io.py
[transform.py]: transform.py
[aggregate.py]: aggregate.py
[score.py]: score.py
[inspect.py]: inspect.py
[norm.csv]: norm.csv

[_tqdm.py]: _tqdm.py
[tqdm]: https://github.com/noamraph/tqdm
