# ML $t \bar{t}$ Idea

Extends naturally reconstruction work in $t \bar{t}$ in the dilepton channel. 
The neutrino kinematics are unknown but can be worked out either in advance using ML methods, or after assignment of the correct $b$-jet--lepton pair.

The principle is to use a feed-forward neural network to assign the correct $b$-jet--lepton pair. Using the truth information we know which $b$-jet comes from the same top-quark as each lepton. We can therefore make two combinations: $b_1 l_1 / b_2 l_2$ and $b_1 l_2 / b_2 l_1$, one of which is the correct combination. By assigning the correct combination a 1 and the incorrect 0, we can train a NN to discriminate between correct and incorrect combinations.

## Technical
* [PyTorch NN tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
* Should likely only need basic `torch` installation.
* Can probably run on own CPU, Google Colab GPU, or CSF gpu.

## Steps
1. Truth-matching (Ethan need to think about this more)
2. Build combinations
3. Convert ROOT file to .h5 (input example)
4. Train model

Similar approach in ATLAS analysis: [https://cds.cern.ch/record/2712338/files/ATL-COM-PHYS-2020-190.pdf?version=9](https://cds.cern.ch/record/2712338/files/ATL-COM-PHYS-2020-190.pdf?version=9) (this link won't work for non-ATLAS memberes)

