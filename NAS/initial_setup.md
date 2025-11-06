# NAS Initial Setup

1. Initial login
1. Setup SSH pass-through
1. Setup Conda configuration
1. Setup HuggingFace configuration

## 1. Initial login
On initial login, you will need to set your system password, login to the Secure Front End (SFE), and login into the Pleiades Front End (PFE). To set your system password, you'll need to contact [NAS user support](https://www.nas.nasa.gov/hecc/support/user_support.html). [This tutorial](https://www.nas.nasa.gov/hecc/support/kb/enabling-your-rsa-securid-soft-token-%28mobile-app%29_538.html#) has the details.

> [!WARNING]
> You must at some point log in to Pleiades Front End (PFE) to create your home directory, which is required to run jobs on NAS.

To log into `pfe` first you must ssh into a secure front end. Replace `username` with your username on NAS. You can also choose a different front end if one is down by changing the 6 in `sfe6` to a number in the range \[6-8\].
```bash
ssh username@sfe6.nas.nasa.gov
```
Then ssh into `pfe` which will create your home directory.
```bash
ssh pfe
```

## 2. SSH Pass-Through
NAS is configured to have a front-end login system, the Secure Front End (SFE), that is then used to SSH into the NAS compute resources, such as the PFE. Setting up SSH pass-through allows you to login into compute resources, such as the PFE, in a single step.

[This tutorial](https://www.nas.nasa.gov/hecc/support/kb/setting-up-public-key-authentication_230.html) describes how to setup public key authentication. Note that if you already have a key that is not password protected, you will need to create a new one that is.

[This tutorial](https://www.nas.nasa.gov/hecc/support/kb/setting-up-ssh-passthrough_232.html) describes how to setup SSH pass-through once public key authentication is setup.

> [!TIP]
> If you are prompted multiple times wait for your RSA token to refresh to a new one. Be careful however — failing more than 3 times will lock your RSA token.

If you run into any issues with your initial setup just call [NAS user support](https://www.nas.nasa.gov/hecc/support/user_support.html) and they will help you work through them. There is also a [troubleshooting](https://www.nas.nasa.gov/hecc/support/kb/common-login-failures-or-issues_162.html) page that has common issues.

## 3. (Optional) Conda settings
NAS comes with pre-built conda environments that can be used for processing (see the full list [here](https://www.nas.nasa.gov/hecc/support/kb/machine-learning-overview_572.html)). If none of the provided environments fit your needs, you can install your own environments after some setup.

**Note that the NAS staff has created the `terramind` conda environment for us. You don't need to follow this setup to use that environment.**

This involves:
1. Pre-loading conda on startup
1. Changing the environment storage paths to a NAS location with adequate storage.

To do this, add the following lines to your `~/.profile` file in PFE:
```bash
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4
export CONDA_INSTALL=/nobackup/$USER/conda/envs
export CONDA_PKGS=/nobackup/$USER/conda/pkgs
conda config --add envs_dirs $CONDA_INSTALL
conda config --add pkgs_dirs $CONDA_PKGS
```
The first two lines load conda on startup, and the next four change the environment and package install directories to locations on the `/nobackup` drive.
