# ML Framework for Extended Higgs-Sector 4Tops

This study looks to see how Higgs boson(s) with different properties from the assumed Standard Model Higgs would manifest, 
and therefore how we can discriminate between a SM Higgs and a non-SM Higgs in 4top production.

We will use a DNN to build a binary classifier which can (hopefully) achieve this discrimination. We will build a DNN framework
and apply it to the case where an additional CP-odd pseudoscalar is introduced. This simple model introduces an enhanced coupling between 
the top quark and the Higgs, $\kappa$, and a mixing between the CP-even SM Higgs and the CP-odd pseudoscalar, characterised by mixing angle $\alpha$.

## Plan
Use parton-level 4top top-quark observables as inputs to a DNN. Train the DNN on simulated samples with and without the effects of additional CP-odd pseudoscalar.

## To Do
1. Consider all the different observables which could be useful e.g. invariant masses, angles between tops, $H_T$; such quantities for various different combinations of the 4tops.
2. Write a Python script to save these observables in a `.h5` file.
3. Write a small `Pytorch` framework which loads the `.h5` data, and builds a classifier.
4. Train the model (probably on a CPU is fine).
5. Visualise outputs: e.g. ROC curve, output distirbution
6. Collect all code together to apply the whole process easily to other such studies.

## Current Use Case
We can probably look at the cases of pure CP-even vs pure CP-odd, and of SM $\kappa$ coupling vs large enhancement to $\kappa$.

## Extensions
* Apply this framework to other extended Higgs sector models
* Consider doing this study using only reco-level observables with no reconstruction
* Extend to more powerful discriminator e.g. GNN
