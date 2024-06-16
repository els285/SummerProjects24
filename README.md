# Summer Projects 24

Git repo for Summer Projects 2024. We will be investigating various ways of improving the discrimination between observed and unobserved Standard Model processes, and improving the kinematic reconstruction for such processes.

<p float="left">
          <img alt="alt_text" width="300px" src="https://user-images.githubusercontent.com/68130081/147861228-f0680d4b-599b-49e3-9afc-c8b58910ed6a.png" />
                       
</p>


## Contact Details

Ethan Simpson - [ethan.simpson@manchester.ac.uk](mailto:ethan.simpson@manchester.ac.uk)           [<img alt="skype_link" width="40px" src="https://private-user-images.githubusercontent.com/68130081/337256068-5669cd38-9f7d-429c-ac7f-f14ee7377456.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTc2NzU5MDIsIm5iZiI6MTcxNzY3NTYwMiwicGF0aCI6Ii82ODEzMDA4MS8zMzcyNTYwNjgtNTY2OWNkMzgtOWY3ZC00MjljLWFjN2YtZjE0ZWU3Mzc3NDU2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA2MDYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNjA2VDEyMDY0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQzYzQzODJhOWE0ZmI3MGQ5OGFlMDcyYjkwYWU2ZmNjZDA1Y2FhZTE4YTQxYjhiMWE3NWIxNjMyOGRmMzRhMGEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.cv4bHwhgS0uJ1WMKEQTgx-nhOBE_yH0lEz95aEBCyec" />](https://join.skype.com/invite/kC5RHVCLuzWU)

Zihan Zhang -  [zihan.zhang@manchester.ac.uk](mailto:zihan.zhang@manchester.ac.uk) Skype name: `live:exapples_3`

Yvonne Peters - [yvonne.peters@manchester.ac.uk](mailto:yvonne.peters@manchester.ac.uk)


## Physics Primer
I wrote a quick physics primer, with overleaf [here](https://www.overleaf.com/read/byxhgqjmqjkw#9112d0)

## Technical Primer
Some computer skills like information about command-line interfaces can be found in [TechnicalPrereqs.md](https://github.com/els285/SummerProjects24/blob/main/TechnicalPrereqs.md)

Possible requirements:
* A place to store and manipulate data: own workshop, CSF, Noether.
* A terminal
* Jupyter notebook environment

### Where/How to work
Best practice is to work in way that makes you feel most comfortable: this will probably be on your own laptop.
Here you can run code in Jupyter / Google Colab notebooks (in a browser), or run scripts from the terminal.
One of the best modern tools is VSCode: [https://code.visualstudio.com/](https://code.visualstudio.com/), which comes itself with a built in terminal, plus you cna build Jupyter notebooks in VSCode which is how I do most code development.

### HEP Software Foundation
HEP software skills: [https://hsf-training.github.io/analysis-essentials/#](https://hsf-training.github.io/analysis-essentials/#)

### HEP data analysis frameworks:
These are used to analyse the data structures we store particle collision information in: reading/writing that data, processing it and applying transformations to it, making histograms and plotting results. 
Particle physics data is generally stored in `.root` files

#### ROOT 
[https://root.cern/](https://root.cern/) - C++ and Python through [PyROOT](https://root.cern/manual/python/). ROOT is the main tool people use to do ATLAS analyses. ROOT installation guide: [https://root.cern/install/#install-via-a-package-manager](https://root.cern/install/#install-via-a-package-manager). ROOT is slightly harder to pick up from a Python background. If you want to stay in particle physics, you will probably have to use "proper ROOT" eventually. 

#### Scikit-HEP 
[https://scikit-hep.org/](https://scikit-hep.org/) - Python-based, "modern" alternative to ROOT. More pythonic syntax. More aligned with "data science" software stack, so arguably more applicable for more general data science. Machine-learning tools in general have to interface to this method.
Short example of doing a quick analysis using Scikit-HEP tools from Andy Pilkington [available here](https://github.com/heppilko/ParticlePhysics-simulation-and-analysis/blob/main/Examples/analysis_python.ipynb). This uses `uproot` to load the ROOT file, uses `vector` to create 4-momentum objects which can manipulated, uses `matplotlib` to create a histogram and plot it (and in the background uses `awkward-arrays` as the array type).


## Simulated Date
CERNbox is a good place to store files: [https://cernbox.cern.ch/s/qQbt4fhHYhQX5Zt](https://cernbox.cern.ch/s/qQbt4fhHYhQX5Zt).

Currently it contains:
* `ttbar_200k_dilep.root` = ROOT file with 200,000 ttbar dilepton events. This sample contains **final-state particles only** (after everything decayed), no top quarks. 
* `ttbar_1M_parton_dilep.root` = ROOT file with 1million ttbar dilepton events. This sample contains the **truth particles**.

## Useful Software Links

[HyPER](https://github.com/tzuhanchang/HyPER)

We will use this to generate the simulated data
[MadLAD](https://github.com/tzuhanchang/MadLAD)
