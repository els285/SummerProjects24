# SMEFT@NLO

Effective field theories are very general and powerful tools which are generic extensions of the SM. 
In these models, we introduce new operators into the SM Lagrangian, which correspond to new interactions in the theory. 
However, there are no new particles, just new vertices!

Each new operator has an associated coefficient (analogous to $kHtt$ or whatever from Higgs Characterisation model) called a "Wilson coefficient". These are in the `param_card.dat`.

SMEFT@NLO is unique (maybe?) in doing loop diagrams but we are just going to look at Leading-Order for now.

Model here:
[https://feynrules.irmp.ucl.ac.be/wiki/SMEFTatNLO](https://feynrules.irmp.ucl.ac.be/wiki/SMEFTatNLO)

Basic model imported via:
```
import model SMEFTatNLO-LO
```
This imports the LO restriction model. You will need to further edit the restriction card (in the same manner as Claudio described) to get rid of all the operators we don't want.
The operators/coefficients we want to keep are numbered are 19,20,23,24,25 in the param card. We will only look at one at a time though so you can either choose to get rid of all but one, or keep them but then have to set the others to zero in the param card when we come to running.

## Running
```
generate p p > t t~ t t~ QED=99 QCD=99 NP=2
```
Try this for now with the restriction card and see how many diagrams it gets.

## Values 
We set each one of the Wilson coefficients to a non-zero number individually (there shouldn't be a scenario where two are non-zero at the same time). We need to do 2 non-zero points for each.
Use the individual bounds from the top of page 16 of this:
[https://arxiv.org/pdf/2404.12809
](https://arxiv.org/pdf/2404.12809).
