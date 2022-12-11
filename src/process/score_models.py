import argparse
import sys
import os
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
from Bio.PDB.PDBParser import PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio import pairwise2
import pdb


parser = argparse.ArgumentParser(description = '''Calculate the loss function towards a reference structure.''')
parser.add_argument('--native_structure', nargs=1, type= str, default=sys.stdin, help = 'Path to pdb with native structure.')
parser.add_argument('--target_id', nargs=1, type= str, default=sys.stdin, help = 'Target id.')
parser.add_argument('--design_df', nargs=1, type= str, default=sys.stdin, help = 'Df containing crop regions and designs.')
parser.add_argument('--pred_dir', nargs=1, type= str, default=sys.stdin, help = 'Directory with predicted structures.')
parser.add_argument('--receptor_chain', nargs=1, type= str, default=sys.stdin, help = 'Receptor chain in native structure')
parser.add_argument('--binder_chain', nargs=1, type= str, default=sys.stdin, help = 'Binder chain in native structure')
parser.add_argument('--mode', nargs=1, type= str, default=sys.stdin, help = 'options: viterbi')
parser.add_argument('--outname', nargs=1, type= str, default=sys.stdin, help = 'Where to write all scores (csv)')

##############FUNCTIONS##############
def read_pdb(pdbname):
    '''Read PDB
    '''

    three_to_one = {'ARG':'R', 'HIS':'H', 'LYS':'K', 'ASP':'D', 'GLU':'E', 'SER':'S', 'THR':'T', 'ASN':'N', 'GLN':'Q', 'CYS':'C', 'GLY':'G', 'PRO':'P', 'ALA':'A', 'ILE':'I', 'LEU':'L', 'MET':'M', 'PHE':'F', 'TRP':'W', 'TYR':'Y', 'VAL':'V',
    'SEC':'U', 'PYL':'O', 'GLX':'X', 'UNK': 'X'}

    parser = PDBParser()
    struc = parser.get_structure('', pdbname)

    #Save
    model_CA_coords = {}
    all_model_coords = {}
    all_model_resnos = {}
    model_seqs = {}
    model_plDDT = {}

    for model in struc:
        for chain in model:
            #Save
            all_model_coords[chain.id]=[]
            model_CA_coords[chain.id]=[]
            all_model_resnos[chain.id]=[]
            model_seqs[chain.id]=[]
            model_plDDT[chain.id]=[]
            #Save residue
            for residue in chain:
                res_name = residue.get_resname()
                if res_name not in three_to_one.keys():
                    continue
                for atom in residue:
                    atom_id = atom.get_id()
                    atm_name = atom.get_name()
                    #Save
                    all_model_coords[chain.id].append(atom.get_coord())
                    all_model_resnos[chain.id].append(residue.get_id()[1])
                    if atom_id=='CA':
                        model_CA_coords[chain.id].append(atom.get_coord())
                        model_seqs[chain.id].append(three_to_one[res_name])
                        model_plDDT[chain.id].append(atom.get_bfactor())



    #Convert to array
    for key in model_CA_coords:
        all_model_coords[key] = np.array(all_model_coords[key])
        model_CA_coords[key] = np.array(model_CA_coords[key])
        all_model_resnos[key] = np.array(all_model_resnos[key])
        model_seqs[key] = np.array(model_seqs[key])
        model_plDDT[key] = np.array(model_plDDT[key])

    return all_model_coords, model_CA_coords, all_model_resnos, model_seqs, model_plDDT

def get_native_features(native_CA_coords, receptor_chain, binder_chain):
    '''Get the native interface residues and centre of mass
    '''

    #Get the CA coords
    receptor_CAs = native_CA_coords[receptor_chain]
    binder_CAs = native_CA_coords[binder_chain]

    #Calc 2-norm - distance between binder and interface
    mat = np.append(receptor_CAs, binder_CAs,axis=0)
    a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    l1 = len(receptor_CAs)
    #Get interface
    contact_dists = dists[:l1,l1:] #first dimension = receptor, second = binder
    contacts = np.argwhere(contact_dists<10) #CAs within 10 Å
    receptor_if_res, binder_if_res = np.unique(contacts[:,0]), np.unique(contacts[:,1])

    #Centre of mass for binder
    COM = np.sum(binder_CAs,axis=0)/(binder_CAs.shape[0])

    return (receptor_if_res, binder_if_res, COM, receptor_CAs, binder_CAs)

def calc_metrics(pred_name, native_receptor_CA_coords, binder_CA_coords,
                native_receptor_if_res, binder_if_res, COM, native_receptor_seq):
    '''Calculate metrics
    average receptor if dists
    average binder if dists
    binder plDDT
    COM distance
    binder RMSD (CA)
    receptor RMSD (CB)
    Interface sequence recovery of receptor
    Overall sequence recovery of receptor
    Contact similarity btw the designed and native binders
    '''

    #Align the native_receptor_seq from the seed and the target
    all_pred_coords, pred_CA_coords,  pred_resnos, pred_seqs, pred_plDDT  = read_pdb(pred_name)
    pred_receptor_seq = ''.join(pred_seqs['A'])
    alignments = pairwise2.align.globalxx(native_receptor_seq, pred_receptor_seq)[0]
    native_aln, pred_aln = alignments[0], alignments[1]
    ni, pi = 0, 0
    keep_native, keep_pred = [], []

    for i in range(len(native_aln)):
        if (native_aln[i]!='-') and (pred_aln[i]!='-'):
            keep_native.append(ni)
            keep_pred.append(pi)
        if native_aln[i]!='-':
            ni+=1
        if pred_aln[i]!='-':
            pi+=1

    #Superpose the receptor CAs and compare the centre of mass
    sup = SVDSuperimposer()

    #Superpose the CA coords of the receptor
    sup.set(native_receptor_CA_coords[keep_native], pred_CA_coords['A'][keep_pred]) #(reference_coords, coords)
    sup.run()
    rot, tran = sup.get_rotran()
    #Rotate the design coords to match the centre of mass for the native comparison
    rotated_coords = np.dot(pred_CA_coords['B'], rot) + tran
    rotated_CM =  np.sum(rotated_coords,axis=0)/(rotated_coords.shape[0])
    delta_CM = np.sqrt(np.sum(np.square(COM-rotated_CM)))

    #Calc 2-norm - distance between peptide and interface
    #Get the interface positions
    pred_if_res = np.unique(pred_resnos['A'])[native_receptor_if_res]
    mat = np.append(all_pred_coords['B'], all_pred_coords['A'][np.isin(pred_resnos['A'], pred_if_res)],axis=0)
    a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    l1 = len(all_pred_coords['B'])
    #Get interface
    contact_dists = dists[:l1,l1:] #first dimension = binder, second = receptor

    #Get the closest atom-atom distances across the receptor interface residues.
    closest_dists_binder = contact_dists[np.arange(contact_dists.shape[0]),np.argmin(contact_dists,axis=1)]
    closest_dists_receptor = contact_dists[np.argmin(contact_dists,axis=0),np.arange(contact_dists.shape[1])]


    return (closest_dists_binder, closest_dists_receptor, pred_plDDT, delta_CM)





#################MAIN####################
def run_scoring(native_structure, seed_df, row_num, preds):
    """Run function for the scoring
    """

    all_native_coords, native_CA_coords,  native_resnos, native_seqs, native_plDDT = read_pdb(native_structure)


    #The target receptor chain is always A and the binder B here
    receptor_chain, binder_chain = 'A', 'B'
    #Get native sequences
    native_receptor_seq = ''.join(native_seqs[receptor_chain])

    #Get receptor if res and binder COM
    (receptor_if_res, binder_if_res, COM, receptor_CAs, binder_CAs) = get_native_features(native_CA_coords, receptor_chain, binder_chain)

    #Go through all crops
    results = {'if_dist_binder':[], 'if_dist_receptor':[], 'plddt':[], 'delta_CM':[] }

    #Get the crop region
    row = seed_df.loc[row_num]
    binder_if_res_crop =  np.argwhere((binder_if_res>=row.cs)&(binder_if_res<row.ce))[:,0]
    binder_if_res_crop-=binder_if_res_crop[0]

    #Get all the predicted files
    n_preds = len(preds)
    #Go through all preds
    for i in range(n_preds):

        #Get the loss metrics
        (closest_dists_binder, closest_dists_receptor,
        pred_plDDT, delta_CM) = calc_metrics(preds[i], receptor_CAs, binder_CAs[row.cs:row.ce],
                                            receptor_if_res, binder_if_res_crop, COM, native_receptor_seq)

        #Save
        results['if_dist_binder'].append(closest_dists_binder.mean())
        results['if_dist_receptor'].append(closest_dists_receptor.mean())
        results['plddt'].append(pred_plDDT['B'].mean())
        results['delta_CM'].append(delta_CM )

        print(i)
    #Df
    results_df = pd.DataFrame.from_dict(results)
    results_df['loss'] = 1/results_df.plddt*results_df.if_dist_binder*results_df.if_dist_receptor*0.5*results_df.delta_CM

    return results_df


#Run
# native_structure='/home/patrick/Desktop/seed_1.pdb'
# seed_df=pd.read_csv('/home/patrick/Desktop/seed_df.csv')
# row_num=0
# preds=['/home/patrick/Desktop/unrelaxed_seed_1_1.pdb']
# run_scoring(native_structure, seed_df, row_num, preds)
