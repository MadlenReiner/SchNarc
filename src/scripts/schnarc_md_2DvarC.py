import sys
import os
import schnarc
import logging
import shutil
import argparse
import numpy as np
import datetime
import copy
import time
(tc, tt) = (time.process_time(), time.time())
#import schnet
import schnetpack as spk
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler
sharc_path = '/user/reiner/sharc/sharc/DEVELOPMENT/pysharc/bin' #bin
sys.path.append(sharc_path)
sys.path.append(os.path.join(sharc_path,'..','lib'))
sys.path.append(os.path.join(sharc_path,'..','lib', 'sharc'))
from sharc.pysharc.interface import SHARC_INTERFACE
#import sharc
#print(sharc.__file__)
import run_schnarc as rs
#import sharc and schnarc
from functools import partial
from schnarc.calculators import SchNarculator, EnsembleSchNarculator
def transform(A,U):
    '''returns U^T.A.U'''

    Ucon=[ [ U[i][j].conjugate() for i in range(len(U)) ] for j in range(len(U)) ]
    B=np.dot(Ucon,np.dot(A,U))

    return B





class SHARC_NN(SHARC_INTERFACE):
    """
    Class for SHARC NN
    """
    # Analytical functions used
    # Name of the interface
    interface = 'NN'
    # store atom ids
    save_atids = True
    # store atom names
    save_atnames = True
    # accepted units:  0 : Bohr, 1 : Angstrom
    iunit = 0
    # not supported keys
    not_supported = ['nacdt', 'dmdr' ]
    

    def initial_setup(self, **kwargs):
        return

    def do_qm_job(self, tasks, Crd):
        """

        Here you should perform all your qm calculations

        depending on the tasks, that were asked
        Hamiltonian matrix and dipole moment matrix are complex
        Gradient and NACs are real
        """
        #QMout = self.QMout
        QMout={}
        # translate Coordinates to xyz and initialise Variables
        x=Crd[0][0]
        y=Crd[0][1]
        z=Crd[0][2]
        A=0.04
        B=0.01
        C=0.5
        D=3.0
        k=0.001
        # calculate hamiltonian
        Hamiltonian=np.zeros((2,2),dtype=np.cdouble)
        #print("here")
        Hamiltonian[0][0]=np.cdouble(A*(x-C)**2+B*(y-D)**2+1.0*z**2 + 0.0000j)
        Hamiltonian[1][1]=np.cdouble(B*(x-D)**2+A*(y-C)**2+1.0*z**2 + 0.0000j)
        Hamiltonian[1][0]=np.cdouble(k*(x+y-2.3) + 0.0000j)
        Hamiltonian[0][1]=np.cdouble(k*(x+y-2.3) + 0.0000j)
        #print("here2")
        d,U=np.linalg.eigh(Hamiltonian)
        DHamiltonian=np.diag(d)
        #print(U.dtype)
        QMout["h"]= np.array(DHamiltonian).tolist()
        # calculate gradiants per surface
        try:
            QMout["overlap"]=np.dot(self.Uold,U)
            #print("set correct overlap")
        except:
            QMout["overlap"]=U
        natom=1
        nmstates=2
        SH2ANA={}
        SH2ANA["gvar"]=[["x","y","z"]]
        deriv={}
        deriv["x"]=[[np.cdouble(A*2*(x-C) + 0.00j),np.cdouble(k + 0.00j)],[np.cdouble(k + 0.00j),np.cdouble(B*2*(x-D) + 0.00j)]]
        deriv["y"]=[[np.cdouble(B*2*(y-D) + 0.00j),np.cdouble(k + 0.00j)],[np.cdouble(k + 0.00j),np.cdouble(A*2*(y-C) + 0.00j)]]
        deriv["z"]=[[np.cdouble(1.0*2*z + 0.0000j),np.cdouble(0.0000 + 0.0000j)],[np.cdouble(0.0000+0.0000j),np.cdouble(1.0*2*z+0.0000j)]]
        QMout["phases"]=[ np.cdouble(1.+0.j) for i in range(nmstates) ]
        for i in range(nmstates):
           if QMout['overlap'][i][i].real<0.:
              QMout['phases'][i]=np.cdouble(-1.+0.j)
        SH2ANA["deriv"]=deriv
        ggrad=[]
        Ucon=[ [ U[i][j].conjugate() for i in range(len(U)) ] for j in range(len(U)) ]
        for iatom in range(natom):
           ggrad.append([])
           for idir in range(3):
              if SH2ANA['gvar'][iatom][idir]=='0':
                  ggrad[-1].append( [ 0. for i in range(nmstates) ] )
              else:
                  v=SH2ANA['gvar'][iatom][idir]
                  A=SH2ANA['deriv'][v]
                  Dmatrix=np.dot(Ucon,np.dot(A,U))
                  #Dmatrix=transform(SH2ANA['deriv'][v],U)
                  ggrad[-1].append( [ Dmatrix[i][i].real for i in range(nmstates) ] )
        # rearrange gradients
        grad=[ [ [ ggrad[iatom][idir][istate] for idir in range(3) ] for iatom in range(natom) ] for istate in range(nmstates) ]
        #print(Dmatrix.dtype)
        QMout["grad"]=grad
        # sharc needs dipole moments here only 0
        QMout["dm"]=np.array(np.zeros((3,2,2))).tolist()
        #QMout = self.schnarc_init.calculate(Crd)
        self.Uold = U
        #print(self.Uold)
        return QMout



    def final_print(self):
        self.sharc_writeQMin()

    def parseTasks(self, tasks):
        """
        these things should be interface dependent

        so write what you love, it covers basically everything
        after savedir information in QMin

        """

        # find init, samestep, restart
        QMin = { key : value for key, value in self.QMin.items() }
        QMin['natom'] = self.NAtoms
        QMin['atname'] = self.AtNames
        self.n_atoms=self.NAtoms
        key_tasks = tasks['tasks'].lower().split()


        if any( [self.not_supported in key_tasks ] ):
            print( "not supported keys: ", self.not_supported )
            sys.exit(16)

        for key in key_tasks:
            QMin[key] = []

        for key in self.states:
            QMin[key] = self.states[key]


        if 'init' in QMin:
            checkscratch(QMin['savedir'])
        if not 'init' in QMin and not 'samestep' in QMin and not 'restart' in QMin:
            fromfile=os.path.join(QMin['savedir'],'U.out')
            if not os.path.isfile(fromfile):
                print( 'ERROR: savedir does not contain U.out! Maybe you need to add "init" to QM.in.' )
                sys.exit(1)
            tofile=os.path.join(QMin['savedir'],'Uold.out')
            shutil.copy(fromfile,tofile)

        for key in ['grad', 'nacdr']:
            if tasks[key].strip() != "":
                QMin[key] = []

        QMin['pwd'] = os.getcwd()

        return QMin

    def readParameter(self, param,  *args, **kwargs):

        self.NNnumber = int(1) #self.options["NNnumber"]
        # Get device
        #self.device = torch.device("cuda" if param.cuda else "cpu")
        self.device = torch.device("cpu")
        self.NAtoms = vars(self)['NAtoms']
        self.AtNames = vars(self)['AtNames']
        self.dummy_crd = np.zeros((self.NAtoms,3))
        #self.hessian = True if param.hessian else False
        self.hessian = False
        #self.nac_approx=param.nac_approx
        self.nac_approx = [1,0.018,0.036]
        #self.schnarc_init = SchNarculator(self.dummy_crd,self.AtNames,param.modelpath,self.device,hessian=self.hessian,nac_approx=self.nac_approx)
        return


def main():
    """
        Main Function if program is called as standalone
    """
    parser = rs.get_parser()
    args = parser.parse_args()
    print( "Initialize: CPU time: % .3f s, wall time: %.3f s"%(time.process_time() - tc, time.time() - tt))
    adaptive = float(1.0)
    param = "schnarc_options"
    # init SHARC_NN class 
    nn = SHARC_NN()
    print( "init SHARC_NN:  CPU time: % .3f s, wall time: %.3f s"%(time.process_time() - tc, time.time() - tt))
    # run sharc dynamics
    print(args)
    nn.run_sharc("input",args,adaptive, initial_param=param)
    #nn.run_sharc("input",args,adaptive, initial_param=param)
    print( "run dynamics:  CPU time: % .3f s, wall time: %.3f s"%(time.process_time() - tc, time.time() - tt))
if __name__ == "__main__":
    main()



