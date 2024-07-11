# DNN + nu^2-Flows Results

We have a test dataset of 226192 events, all identified by their eventNumber.

Your DNN work shoudl output the correct $b1,l^+,b2,l^-$ kinematics.

The file `callum_neutrinos.h5` has the neutrino kinematics, with the corresponding eventNumber (should be in the same order as your test.h5 file unless Nu2-Flows is really messing stuff up).
The `neutrinos` dataset has structure `shape (226192, 2, 3)` where the first element corresponds to the event, the second element to the top or antitop, with top coming first, and the third element to the three-momentum $p_x,p_y,p_z$ of each neutrino.
For example, the object `(0,0,:)` will be the numpy array of the 3-momentum of the top quark of the first event.
The object `(0,1,:)` will be the numpy array of the 3-momentum of the anti-top quark of the first event.

## Task 1
Build the neutrino 4-momentum `vectors` from their 3-momentum by setting mass to zero.
Build the 4-momentum of the top and antitop and ttbar, as before, using the correct neutrino and the correct $b$ and $l$ from your network.
Compare this to the truth top, antitop and ttbar kinematics which are given in file `truth_skimmed_july9_v2.root` (you will need to match the eventNumbers to get the correct events from this).

## Task 2
Repeat this, but instead of using your DNN results, just randomly assign each $b$-jet to the top and antitop. This should give worse results...
**Outcome:** Compare your DNN to random assignment.

## Task 3 
Using the `EM_nu`,`SM_nu` and `NW_nu` branches of the `recon` tree in `truth_skimmed_july9_v2.root`, build the corresponding neutrinos for these three other methods (these are other techniques to build the neutrino 3-momentum, and we comparing to nu^2-Flows).
Build the top kinematics, and compare to Task 1, where in each case you use the $b$ and $l$ kinematics from your DNN.
**Outcome:** Comapre Nu2-Flows to other techniques.

The most useful comparison is to make a histogram of the difference between a kinematic and truth:
<img width="542" alt="image" src="https://github.com/els285/SummerProjects24/assets/68130081/9b1473ef-ead6-42bd-acb7-7233fee8d6ea">

e.g.
```python
top_truth.pt - top_nu2.pt
```

Be good to have all of this as a script rather than a Jupyter Notebook in the end!
