# Machine Learning Stuff 

Modern trends seem to be moving towards `PyTorch`, which is what HyPER is built around for example, so we should focus on that, but the skills are transferrable. 

##  Tutorials
All in a HEP context...

### HSF ML Tutorial
Here is the official HSF tutorial which I think gives a good introduction to classification using PyTorch (it uses `scikit-learn` first which is a good easier intro):
[https://hsf-training.github.io/hsf-training-ml-webpage/]https://hsf-training.github.io/hsf-training-ml-webpage/)

You are meant to run this in a Kaggle Jupyter notebook by making a Kaggle account (maybe this is of intrest to you anyway).
The tutorial will definitely work that way.
If you want to do it on your own machine, give it a try: I've uploaded the datasets I think you need to the Google Drive in [this subdirectory](https://drive.google.com/drive/folders/1MXqiK9E-uVTh8K2_Y8-r98InRAQBUrvJ).

### Other Tutorials
Many other cool tutorials from a Summer School I didn't get to go to because of Covid (more complex):
[https://github.com/makagan/SSI_Projects/tree/main](https://github.com/makagan/SSI_Projects/tree/main).
In particular, I would highlight:
* [Advanced Python](https://github.com/makagan/SSI_Projects/blob/main/python_basics/python_intro_part2.ipynb) (if you're not familiar with classes etc):
* [Intro to PyTorch](https://github.com/makagan/SSI_Projects/blob/main/pytorch_basics/pytorch_intro.ipynb)
* [Building neural networks with PyTorch](https://github.com/makagan/SSI_Projects/blob/main/pytorch_basics/pytorch_intro.ipynb)

[MLHEP School tutorials
]([url](https://github.com/MLHEP-school/2023/tree/main))
## Converting ROOT data
The default file format for machine learning is `.h5`, so the first thing one
has to do is convert ROOT files to this format. A very short example notebook is
included `example_ROOT2h5.ipynb`.
