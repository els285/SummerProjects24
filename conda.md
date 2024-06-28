# Useful CSF Stuff

## VSCode
VSCode can be made to work on CSF, following [these instructions](https://ri.itservices.manchester.ac.uk/csf3/software/tools/vscode/). You require the `Remote SSH` extension in VSCode.
However, I haven't managed to get Jupyter notebooks working in VSCode on CSF.

## Conda/Miniforge
USe `miniforge` to create conda environments with pre-requisites installed. 
This is especially useful for HyPER which relies on particular torch libraries which can be problematic to install.

Installation of miniforge found here: [Miniforge instructions](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install).

### Main Commands
Building a conda environment from a YAML file:
```bash
conda env create -f yamlconfig.yml
```
To then activate an environment with name `MyEnv`:
```bash
conda activate MyEnv
```

## Batch Running
To access GPUs and to not worry about your ssh connection dying, use a batch system to schedule the running of "jobs". 
This means running some script remotely. It requires an `qsub` submssion script, for example:
```
#!/bin/bash --login
#$ -cwd                # Job will run in the current directory (where you ran qsub)
#$ -o ./logs           # Write the outputs to the logs directory
#$ -l nvidia_a100=<M>  # Runs on M GPU nodes
#$ -pe smp.pe <N>      # Runs on N CPU nodes


echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"

# Load required environment
__conda_setup="$('/mnt/eps01-rds/atlas_lhc/share/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/eps01-rds/atlas_lhc/share/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/mnt/eps01-rds/atlas_lhc/share/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/eps01-rds/atlas_lhc/share/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/mnt/eps01-rds/atlas_lhc/share/miniforge3/etc/profile.d/mamba.sh" ]; then
    . "/mnt/eps01-rds/atlas_lhc/share/miniforge3/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<
conda activate <conda_env>
echo $CONDA_PREFIX

# Copy input datasets to TMPDIR for speed
cp <path_to_input_dataset/dataset> $TMPDIR

# Next build command to run python training script
command="source train_JetClass.sh PELICAN kin"

# Run command
echo "================================"
echo "Will run command ${command}"
$command
echo -e "\nDone!"
```
