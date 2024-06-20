# ML $t \bar{t}$ Idea

Extends naturally reconstruction work in $t \bar{t}$ in the dilepton channel. 
The neutrino kinematics are unknown but can be worked out either in advance using ML methods, or after assignment of the correct $b$-jet--lepton pair.

The principle is to use a feed-forward neural network to assign the correct $b$-jet--lepton pair. Using the truth information we know which $b$-jet comes from the same top-quark as each lepton. We can therefore make two combinations: $b_1 l_1 / b_2 l_2$ and $b_1 l_2 / b_2 l_1$, one of which is the correct combination. By assigning the correct combination a 1 and the incorrect 0, we can train a NN to discriminate between correct and incorrect combinations.

## Technical
* [This tutorial looks nice and brief](https://www.python-engineer.com/courses/pytorchbeginner/13-feedforward-neural-network/)
* [PyTorch NN tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
* Should likely only need basic `torch` installation.
* Can probably run on own CPU, Google Colab GPU, or CSF gpu.

## Steps
1. Truth-matching: $\Delta R$ match the $b$-jets.
     Load the `reco` tree and the `particleLevel` trees as separate arrays, build four-vectors for leptons and jets, then work out for   each event if each particleLevel-jet is within $\Delta R < 0.4$ of a reco-jet, and each particleLevel-lepton is within $\Delta R <     0.1$ of a reco-lepton.
   Pseudo-code:
   ```
     i. Compute the deltaR for each pair of truth-jets and reco-jets: e.g. DeltaR(truth_jet,reco_jet)
     ii. We want 2 jets, so select the two candidates have smallest deltaRs
     iii. Repeat for all events
   ```
   
3. Build some combinations of observables including $m_{lb}$ , $\Delta R(l,b)$ etc.
4. Decide on all inputs to include in training
5. Convert ROOT file to .h5 (see example)
6. Train model... more to follow.

Similar approach in ATLAS analysis: [https://cds.cern.ch/record/2712338/files/ATL-COM-PHYS-2020-190.pdf?version=9](https://cds.cern.ch/record/2712338/files/ATL-COM-PHYS-2020-190.pdf?version=9) (this link won't work for non-ATLAS members)

