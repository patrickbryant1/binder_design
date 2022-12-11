import argparse
import sys
import os
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
from Bio.PDB.PDBParser import PDBParser
import time
import pdb


##############FUNCTIONS##############
def read_pdb(pdbname):
    '''Read PDB
    '''

    three_to_one = {'ARG':'R', 'HIS':'H', 'LYS':'K', 'ASP':'D', 'GLU':'E', 'SER':'S', 'THR':'T', 'ASN':'N', 'GLN':'Q', 'CYS':'C', 'GLY':'G', 'PRO':'P', 'ALA':'A', 'ILE':'I', 'LEU':'L', 'MET':'M', 'PHE':'F', 'TRP':'W', 'TYR':'Y', 'VAL':'V',
    'SEC':'U', 'PYL':'O', 'GLX':'X', 'UNK': 'X'}

    parser = PDBParser()
    struc = parser.get_structure('', pdbname)

    #Save
    model_coords = {}
    model_seqs = {}
    model_atoms = {}
    model_resnos = {}

    for model in struc:
        for chain in model:
            #Save
            model_coords[chain.id]=[]
            model_seqs[chain.id]=[]
            model_atoms[chain.id]=[]
            model_resnos[chain.id]=[]

            #Save residue
            for residue in chain:
                res_name = residue.get_resname()
                if res_name not in three_to_one.keys():
                    continue
                for atom in residue:
                    atom_id = atom.get_id()
                    atm_name = atom.get_name()
                    x,y,z = atom.get_coord()
                    #Save
                    model_coords[chain.id].append(atom.get_coord())
                    model_seqs[chain.id].append(res_name)
                    model_atoms[chain.id].append(atom_id)
                    model_resnos[chain.id].append(residue.get_id()[1])



    #Convert to array
    for key in model_coords:
        model_coords[key] = np.array(model_coords[key])
        model_seqs[key] = np.array(model_seqs[key])
        model_atoms[key] = np.array(model_atoms[key])
        model_resnos[key] = np.array(model_resnos[key])

    return model_coords, model_seqs, model_atoms, model_resnos


def load_model():
    """Load the ESM-IF1 model
    """
    import esm
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    return model

def design_seqs(n_seqs_per_seed, ch1_coords, ch2_coords, ch1_atoms, ch2_atoms, ch1_seq, ch2_seq, model=None, t=1e-6):
    """Design seqs for the provided coords
    """

    #Load the model
    model = load_model()
    #Get the backbone coords (3x3 matrix for each residue; N, CA, C coords
    N, CA, C = ch1_coords[np.argwhere(ch1_atoms=='N')[:,0]], ch1_coords[np.argwhere(ch1_atoms=='CA')[:,0]], ch1_coords[np.argwhere(ch1_atoms=='C')[:,0]]
    ch1_bb_coords = np.concatenate([np.expand_dims(N,1), np.expand_dims(CA,1), np.expand_dims(C,1)], axis=1)
    #Keep the ch1_bb_coords maxx 200 res
    ch1_bb_coords = ch1_coords[:200]
    N, CA, C = ch2_coords[np.argwhere(ch2_atoms=='N')[:,0]], ch2_coords[np.argwhere(ch2_atoms=='CA')[:,0]], ch2_coords[np.argwhere(ch2_atoms=='C')[:,0]]
    ch2_bb_coords = np.concatenate([np.expand_dims(N,1), np.expand_dims(CA,1), np.expand_dims(C,1)], axis=1)

    #Seqs
    three_to_one = {'ARG':'R', 'HIS':'H', 'LYS':'K', 'ASP':'D', 'GLU':'E', 'SER':'S', 'THR':'T', 'ASN':'N', 'GLN':'Q', 'CYS':'C', 'GLY':'G', 'PRO':'P', 'ALA':'A', 'ILE':'I', 'LEU':'L', 'MET':'M', 'PHE':'F', 'TRP':'W', 'TYR':'Y', 'VAL':'V',
    'SEC':'U', 'PYL':'O', 'GLX':'X', 'UNK': 'X'}
    ch1_seq = ch1_seq[np.argwhere(ch1_atoms=='CA')[:,0]]
    ch1_seq = [three_to_one[x] for x in ch1_seq]
    ch1_seq = ''.join(ch1_seq)
    ch2_seq = ch2_seq[np.argwhere(ch2_atoms=='CA')[:,0]]
    ch2_seq = [three_to_one[x] for x in ch2_seq]
    ch2_seq = ''.join(ch2_seq)

    #Create a span of masked residues:
    #To partially mask backbone coordinates, simply set the masked coordinates to np.inf.
    gap = np.zeros((10,3,3))
    gap[:] = np.inf

    #Go through the crops
    designed_receptor_seq, designed_binder_seq = [], []
    native_receptor_seq, native_binder_seq = [], []
    #The design chain is put second with a 10 residue masked region in btw the target
    #The reason for this is to avoid excessive methionines
    #Design towards chain A - crop B
    design_coords = np.concatenate([ch1_bb_coords, gap, ch2_bb_coords])
    native_rsec, native_bseq = ch1_seq, ch2_seq
    design_size = len(ch2_seq)
    #Loop and design
    designed_binder_seq = []
    for i in range(n_seqs_per_seed):
        #Run ESM-IF1
        t0 = time.time()
        sampled_seq = model.sample(np.array(design_coords, dtype=np.float32), temperature=t)
        t1 = time.time()
        print('Sampling took', t1-t0, 's for a size of', len(ch1_seq)+len(ch2_seq)+len(gap))
        #Design
        designed_binder_seq.append(sampled_seq[-design_size:])

    #Add to df
    design_df = pd.DataFrame()
    design_df['designed_binder_seq']=designed_binder_seq

    return design_df
