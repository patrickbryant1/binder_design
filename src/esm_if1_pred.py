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

parser = argparse.ArgumentParser(description = '''Parse PDB files and get interacting chains.''')

parser.add_argument('--crop_df', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with crops of interacting chains.')
parser.add_argument('--n_seqs_per_crop', nargs=1, type= int, default=sys.stdin, help = 'How many sequences to generate per crop.')
parser.add_argument('--structure', nargs=1, type= str, default=sys.stdin, help = 'Path to PDB file.')
parser.add_argument('--temperature', nargs=1, type= float, default=sys.stdin, help = 'Temperature for sequence sampling.')
parser.add_argument('--crop_len', nargs=1, type= int, default=sys.stdin, help = 'What crop length to use.')
parser.add_argument('--outname', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

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

def design_seqs(crop_df, n_seqs_per_crop, ch1_coords, ch2_coords, ch1_atoms, ch2_atoms, ch1_seq, ch2_seq, model=None, t=1e-6, crop_len=0):
    """Design seqs for the provided coords
    """

    #Get the backbone coords (3x3 matrix for each residue; N, CA, C coords
    N, CA, C = ch1_coords[np.argwhere(ch1_atoms=='N')[:,0]], ch1_coords[np.argwhere(ch1_atoms=='CA')[:,0]], ch1_coords[np.argwhere(ch1_atoms=='C')[:,0]]
    ch1_bb_coords = np.concatenate([np.expand_dims(N,1), np.expand_dims(CA,1), np.expand_dims(C,1)], axis=1)
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

    #Select crops
    crop_df['Length']=crop_df.ce-crop_df.cs
    if crop_len>0:
        crop_df = crop_df[crop_df.Length==crop_len]

    #Create a span of masked residues:
    #To partially mask backbone coordinates, simply set the masked coordinates to np.inf.
    gap = np.zeros((10,3,3))
    gap[:] = np.inf

    #Go through the crops
    designed_receptor_seq, designed_binder_seq = [], []
    native_receptor_seq, native_binder_seq = [], []
    for ind, row in crop_df.iterrows():
        #The design chain is put second with a 10 residue masked region in btw the target
        #The reason for this is to avoid excessive methionines
        #Design towards chain B - crop A
        if row.Chain=='A':
            design_coords = np.concatenate([ch2_bb_coords, gap, ch1_bb_coords[row.cs:row.ce]])
            native_rsec, native_bseq = ch2_seq, ch1_seq
        #Design towards chain A - crop B
        else:
            design_coords = np.concatenate([ch1_bb_coords, gap, ch2_bb_coords[row.cs:row.ce]])
            native_rsec, native_bseq = ch1_seq, ch2_seq

        designed_receptor_seq_crop, designed_binder_seq_crop = [], []
        native_receptor_seq_crop, native_binder_seq_crop = [], []
        for i in range(n_seqs_per_crop):
            #Run ESM-IF1
            t0 = time.time()
            sampled_seq = model.sample(np.array(design_coords, dtype=np.float32), temperature=t)
            t1 = time.time()
            design_size = row.ce-row.cs
            print('Sampling took', t1-t0, 's for a size of', design_size+len(native_rsec))
            #Design
            designed_receptor_seq_crop.append(sampled_seq[:-(design_size+gap.shape[0])])
            designed_binder_seq_crop.append(sampled_seq[-design_size:])
            #Native
            native_receptor_seq_crop.append(native_rsec)
            native_binder_seq_crop.append(native_bseq[row.cs:row.ce])
        #Save all samples
        designed_receptor_seq.append('-'.join(designed_receptor_seq_crop))
        designed_binder_seq.append('-'.join(designed_binder_seq_crop))
        #Native
        native_receptor_seq.append('-'.join(native_receptor_seq_crop))
        native_binder_seq.append('-'.join(native_binder_seq_crop))

    #Add to df
    crop_df['designed_receptor_seq']=designed_receptor_seq
    crop_df['native_receptor_seq']=native_receptor_seq
    crop_df['designed_binder_seq']=designed_binder_seq
    crop_df['native_binder_seq']=native_binder_seq

    return crop_df
##################MAIN#######################

#Parse args
args = parser.parse_args()
#Data
crop_df = pd.read_csv(args.crop_df[0])
n_seqs_per_crop = args.n_seqs_per_crop[0]
model_coords, model_seqs, model_atoms, model_resnos = read_pdb(args.structure[0])
temperature = args.temperature[0]
crop_len = args.crop_len[0]
outname = args.outname[0]
#Get model
model = load_model()
#Design seqs
crop_df = design_seqs(crop_df, n_seqs_per_crop, model_coords['A'], model_coords['B'],
            model_atoms['A'], model_atoms['B'],
            model_seqs['A'], model_seqs['B'],
            model, temperature, crop_len)

#Save
crop_df.to_csv(outname, index=None)
