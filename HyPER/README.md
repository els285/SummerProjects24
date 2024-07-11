# HyPER

Run on CSF.
Create the `PYG` conda environment:
```bash
conda env create -f hyper_environment.yaml
```

Have the `pyg_setup.sh` script in the same directory as where you submit your HyPER job.

Run
```bash
qsub hyper.sub
```
