# Technical Prerequisites

This is an evolving list of technical (i.e. non-physics) information which I think will be useful. Unfortunately as well as knowing lots of physics you also have to have a pretty broad knowledge of coding, installing and compiling software, UNIX platforms etc.

## How we do physics
In my experience, most physical science research nowadays involved developing or using various different computer programmes, including writing one's own instructions for the computer.
Really the only way to do this is to have a command-line-interface (CLI): some kind of "terminal" which allows us to write text-based instructions which tell the computer what to do.
Nearly everybody uses UNIX-based operating systems, which is why you will see everyone in the HEP department generally using Macs or Linux-based operating systems: it's just easier to do everything.
This is what a terminal might look like...

<img width="600" alt="image" src="https://github.com/els285/SummerProjects24/assets/68130081/1489734c-1d13-4a81-9562-68588cf03117">

In this, I made a new directory, moved in to that directory, opened a text editor called `nano` to create a new python script, ran that script (the output was the sentence *"Is the terminal a blessing or a curse"*), then printed the contents of the directory, then went back to the original directory. (When you open a terminal, you won't see the fancy colour scheme I've set up in mine, sorry...)

## For Windows

Windows has as terminal called `PowerShell`, which is the same idea.
However, it's probably best to set up a Linux environment, and I think the best solution is `Windows Subsystem for Linux` (looks a lot easier than trying to set such things up on Windows a decade ago...). 
This looks like a good video to follow to set it up: [WSL Instructions](https://www.youtube.com/watch?v=qYlgUDKKK5A)


## Using the Terminal

...is fiddly at first. There are specific commands that do particular things, like copy files, make folders, run python etc. You kinda just have to learn these, for example at
[Useful linux commands](https://www.hostinger.co.uk/tutorials/linux-commands)

### Running Python in the terminal
[Verbose explanation here](https://vteams.com/blog/how-to-run-a-python-script-in-terminal/)

Running a Python **script** from the terminal is easy. You put all your Python code into a file and save that file with a name ending in `.py` e.g. `my_script.py`. You then run it (on unix-based system) by simply calling `python` or `python3`:
```bash
python my_script.py
```
**Aside on knowing what verison of Python you are running**
From the terminal, you can do
```bash
python --version
```
to find out which verison you are using, and
```bash
which python
```
to find out the path of the python executable (or more generally any executable).




## CSF

`CSF` is a shared computing facility for research at UoM. It provides us with access to shared disk space and GPUs for training ML models.
To access thse facilities, I've set up accounts for you. We access through the `ssh` protocol.

CSF message:
Hi, I have set up your IT username on the CSF3 (Computational Shared Facility). We have a detailed introduction at: https://ri.itservices.manchester.ac.uk/csf3/getting-started/ This includes: * how to log in * filesystem availability, usage and policy * accessing software * a 10 minute tutorial about batch (recommended, all work must be submitted to this) Other tabs on the website lead to more detailed information on these topics and available software. No limits are imposed on the amount of CPU work you can run. However, the system can get busy so it is best to submit work as and when you have it ready to run. We have a lot of software applications already installed but we may not have the same version number of the software you have been using on your PC/laptop. Please try out an already-installed version before asking for a new install as it can take a couple of weeks or so to make new software available. To see a list of installed software, or, if you would like to try and install software in your home directory please follow the guidance here: https://ri.itservices.manchester.ac.uk/csf3/software/#Installing_your_own_Software Home disk space is very limited. You should keep temporary files in the scratch filesystem and also run your jobs from there. Please note that scratch is not for keeping files in long term and we operate an automated file deletion policy on it (see the filesystems tab on the website). If you need to safely store a large amount of data please check with your PI/Supervisor if they any Research Data Storage. If they do not they can apply for some and request it be accessible on the CSF. See the RDS website for more info: https://ri.itservices.manchester.ac.uk/rds/ Please note that the CSF should not be used for processing sensitive information. Please acknowledge your use of the CSF in any presentation, papers etc using the suggested text at: https://ri.itservices.manchester.ac.uk/csf3/overview/ack/ as it helps us ensure sustainability of the service. Kind regards,

### SSH 
This is a way of connecting to remote networks from your own terminal. I use it all the time to connect to cern computing infrastruture for example. 
You can ssh to CSF via the following command:
```
ssh <username>@csf3.itservices.manchester.ac.uk
```
upon which you'll then be prompted for a password and possible two-factor authentication for log in.

### VPN Access
For security reasons, you can only access CSF if you are using UoM wifi, not eduroam or any other network.
We can get around this by setting up a VPN, with instructions here: [https://www.itservices.manchester.ac.uk/ourservices/popular/vpn/](https://www.itservices.manchester.ac.uk/ourservices/popular/vpn/).





