# SchNarc

A SchNetPack/SHARC interface for machine-learning accelerated excited state simulations.

# Installation and installation requirements on Linux operating systems

Many roads lead to Rome and there are many ways to install programs under linux, especially when using different variants of compiler optimizations. The following is a simplistic route to installing SchNarc. Installation time is expected to take about 30-60 min on a "normal" desktop computer.

You need a python installation with version 3.5 or later.  
We recommend installing Anaconda with python 3 (see https://www.anaconda.com/distribution/).  
If a package that you need, cannot be found, you can use different channels with the option -c or add channels (in this example conda-forge) with:  
``conda config --append channels conda-forge``  
It is recommended to create an environment (in this example the environment is called ml) with:  
``conda create -n ml python h5py tensorboardX pytorch ase numpy six protobuf scipy matplotlib python-dateutil pyyaml tqdm pyparsing kiwisolver cycler netcdf4 hdf5 h5utils gfortran_linux-64``  
Then activate the environment:  
``conda activate ml``   

Install SHARC with pysharc (see https://sharc-md.org/?page_id=50#tth_sEc2.3 or follow the instructions below; version 2.1) in a suitable folder
(``<some-path>``; inside this folder, git will create automatically a folder called sharc):  
``cd <some-path>``  
``git clone https://github.com/sharc-md/sharc.git``  
``cd sharc/source``  
Edit Makefile and make the following changes:  
``    USE_PYSHARC := true``  
``    USE_LIBS := mkl``  
``    ANACONDA := <path/to/anaconda>/anaconda3/envs/ml``  
``    MKLROOT := ${ANACONDA}/lib``  
``  #CC :=gcc``  
``  #F90 :=gfortran``  
``  LD= -L$(MKLROOT) -lmkl_rt -lpthread -lm -lgfortran $(NETCDF_LIB)``  
i.e., delete ``/lib/intel64`` after ``-L$(MKLROOT)``. The 4th change is a line that needs to be added (about MKLROOT). The 5rd and 6th change mean that you have to comment out the definition of CC and F90 and rather use the CC and F90 variables provided by the environment, which is set by anaconda to something like ``<your-anaconda-path>/x86_64-conda_cos6-linux-gnu-cc`` instead of gcc.  

Got to the pysharc/sharc folder:
``cd ../pysharc/sharc`` 
Edit \_\_init\_\_.py  there and make the following changes:  
``    #import sharc as sharc``

Go to the pysharc/netcdf folder:  
``cd ../netcdf``  
Edit Makefile  there and make the following changes:  
``    ANACONDA := <path/to/anaconda>/anaconda3/envs/pysharc``  
``    #CC=$(ANACONDA)/bin/gcc``  
Then go to the pysharc folder and run the installation procedure:  
``cd ..``  
``make install``  
Afterwards, go to the source folder and run the installation procedure:  
``cd ../source``  
``make install``  

Install SchNet (v0.3) in a suitable folder:  
``cd <some-path>``  
``git clone https://github.com/atomistic-machine-learning/schnetpack.git``  
Then go to the directory schnetpack 
``cd schnetpack``  
and carry out:  
``pip install .`` 

If you haven't done so, get the SchNarc sources:  
``git clone https://github.com/schnarc/schnarc.git``  
Then go to the directory schnarc  
``cd schnarc``  
and carry out:  
``pip install .``  

Training or running works in the same way, SchNet works, have a look at https://github.com/atomistic-machine-learning/schnetpack

A tutorial is provided. For this purpose, please download the pdf "SchNarc_Tutorial.pdf" and the corresponding "Tutorial.zip" file, which is automatically downloaded with the code. Follow the instructions provided in the pdf file. A tutorial including exemplary files to train SchNarc on permanent and transition dipole moments to predict UV/vis absorption spectra and latent ML charges can be found at 10.6084/m9.figshare.14832396.v1 (https://figshare.com/articles/dataset/BookChapter_ExcitedStatePropertyLearning_CaseStudy2/14832396). We recommend to switch to the "DipoleMoments_Spectra" branch to follow the tutorial.
