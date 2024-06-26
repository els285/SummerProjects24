# Scikit-HEP (and other useful python modules) thoughts

Remember that if you don't have a module installed, you can pip-install it from the CLI `pip install the-module`, and in a Jupyter/Colab/VScode notebook by prepending an exclamation mark.
(Better still from CLI is to install to a specific python version e.g. `python3.10 -m pip install the-module`

## Manipulating the data
`uproot` for reading and writing ROOT files.

`awkward` is probably the best tool to work with particle physics data. You can apply many of the same type of operations as Numpy.

`numpy` is ubiquitous array tool for Python. Sometime we have to revert to numpy arrays, for example when saving to file formats useful to machine learning.

`pandas` is a dataframe package which gives mroe structure to arrays. Can be useful in specific cases for particle physics, but not the most natural format.

## Making Histograms
I like `boost-histogram` for making histograms with more features than `matplotlib` histograms:
```python
import boost_histogram as bh
hist1 = bh.Histogram(bh.axis.Regular(50, 0, 500)) # Initialises empty histogram with 50 bins spanning [0,500]
hist1.fill(some_data_array)    # Fills the histogram with some data
```

## Plotting 
`mplhep` is a wrapper for `matplotlib.pyplot` which gives "better" particle physics plots:
```python
# Using the histogram from above
import mplhep as hep
import matplotlib.pyplot as plt

hep.histplot(hist1) # Creates an instances of plt with the histogram
plt.show()   # Shows the figure
plt.savefig("filenamme.pdf",dpi=300) # Saves the figure with a resolution of 300 dots per inch
```
