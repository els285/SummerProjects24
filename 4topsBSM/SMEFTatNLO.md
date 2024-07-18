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
These are the ones we do want (see if you can work out which Wilson coeffcients these pertain to in the param card):
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/6cc1c7fe-b990-4c71-8526-d095d91eff77/1163a4eb-28b3-409c-99cf-cbeb2dfc7fd6/Untitled.png)
