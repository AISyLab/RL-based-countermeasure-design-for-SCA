Reinforcement Learning for Profiled Side-Channel Analysis
========
The code for the experiments generating countermeasure combinations can be found in the `countermeasures` folder.
The code for the a posteriori application of the `clock jitter` and `random delay interrupt` countermeasures can be found in the `cython_extensions` folder.
The definitions for the experiments are in the `cm_models` folder.
To generate an overview of the results and to create the scatter plots, use `python -m countermeasures.display_results -h` for instructions.

The cython package source for some of the countermeasures can be found in the `cython_extensions` folder. For instructions on how to build and install that package, see the README.md located there.


Scripts to run most of the experiments can be found in the current folder.
requirements.txt includes all requirements including the exact dependencies used, while requirements.minimal.txt only includes the explicitly installed requirements (generated with `pip-chill`)

**NB:** at most tensorflow 2.1, cuda 10.1 and cudnn 10.1-7.6.0.64 are required. Higher tensorflow versions have some breaking changes that cause the code crash (mostly the One-Cycle LR code). 
**NB:** If running on the TU Delft HPC: make sure to use tensorflow 2.0, cuda 10.1 and cudnn 10.0-7.6.0.64, because of dynamic linking issues concerning some library locations that changed in cuda and cudnn 10.1. Tensorflow makes some assumptions that do not hold true on the HPC and this setup is not supported by the cluster admin. This also requires at most python 3.7, as there is no tensorflow 2.0.0 build for python 3.8+.
#
**This code is based on:**

**[Designing Neural Network Architectures Using Reinforcement Learning](https://arxiv.org/pdf/1611.02167.pdf)**   
Bowen Baker, Otkrist Gupta, Nikhil Naik, Ramesh Raskar  
*International Conference on Learning Representations*, 2017

The source code of which can be found on [Github](https://github.com/bowenbaker/metaqnn/) under the MIT License.
