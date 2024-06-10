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

## CSF

`CSF` is a shared computing facility for research at UoM. It provides us with access to shared disk space and GPUs for training ML models.
To access thse facilities, I've set up accounts for you. We access through the `ssh` protocol.

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





