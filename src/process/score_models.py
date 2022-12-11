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
    model_CB_coords = {}
    all_model_coords = {}
    all_model_resnos = {}
    model_seqs = {}
    model_plDDT = {}

    for model in struc:
        for chain in model:
            #Save
            all_model_coords[chain.id]=[]
            model_CA_coords[chain.id]=[]
            model_CB_coords[chain.id]=[]
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
                    if atom_id=='CB' or (atom_id=='CA' and res_name=='GLY'):
                        model_CB_coords[chain.id].append(atom.get_coord())




    #Convert to array
    for key in model_CA_coords:
        all_model_coords[key] = np.array(all_model_coords[key])
        model_CA_coords[key] = np.array(model_CA_coords[key])
        model_CB_coords[key] = np.array(model_CB_coords[key])
        all_model_resnos[key] = np.array(all_model_resnos[key])
        model_seqs[key] = np.array(model_seqs[key])
        model_plDDT[key] = np.array(model_plDDT[key])

    return all_model_coords, model_CA_coords, model_CB_coords, all_model_resnos, model_seqs, model_plDDT

def get_native_features(native_CA_coords, native_CB_coords, receptor_chain, binder_chain, native_binder_seq):
    '''Get the native interface residues and centre of mass
    '''

    #Get the CA coords
    receptor_CAs = native_CA_coords[receptor_chain]
    binder_CAs = native_CA_coords[binder_chain]
    #Get the CB coords
    #receptor_CBs = native_CB_coords[receptor_chain]
    #binder_CBs = native_CB_coords[binder_chain]

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

    return (receptor_if_res, binder_if_res, COM, receptor_CAs, binder_CAs, contacts)

def calc_if_rmsd(native_CA, pred_CA):
    """Calculate the RMSD of the CA residues
    in the interface btw the designed and native peptide.

    The interface is defined as Cβs within 8 Å between the peptide and its receptor protein,
    and the interface RMSD is calculated after aligning the CAs between the predicted and
    native receptor structures.
    """

    mat = np.append(native_CA, pred_CA,axis=0)
    a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    l1 = len(native_CA)
    contact_dists = dists[:l1,l1:]
    CA_rmsd = np.diagonal(contact_dists)
    return CA_rmsd.mean()

def calc_target_if_seq_recovery(pred_seq, true_seq, if_pos):
    """Calculate the sequence recovery in the target residues
    """

    pred_if_seq = ''.join(pred_seq[if_pos])
    true_if_seq = ''.join(true_seq[if_pos])
    if_recovery = np.mean([(a==b) for a, b in zip(pred_if_seq, true_if_seq)])
    overall_recovery = np.mean([(a==b) for a, b in zip(''.join(pred_seq), ''.join(true_seq))])
    return if_recovery, overall_recovery

def group_residues(contacts, binder_seq):
    """Group residues to analyse contacts
    int_res is a list of interacting residues to each
    of the target residues in the receptor interface
    """

    #Order the contacts
    ordered_contacts = {}
    for i in range(len(contacts)):
        if contacts[i,0] in ordered_contacts:
            ordered_contacts[contacts[i,0]].append(binder_seq[contacts[i,1]])
        else:
            ordered_contacts[contacts[i,0]] = [binder_seq[contacts[i,1]]]

    #Group the receptor_binder_contacts
    hp = 1
    small = 2
    polar = 3
    neg = 4
    pos = 5

    AA_groups = { 'A':hp,'R':pos,'N':polar,'D':neg,'C':polar,'E':neg,
                    'Q':polar,'G':small,'H':pos,'I':hp,'L':hp,'K':pos,
                    'M':hp,'F':hp,'P':hp,'S':polar,'T':polar,'W':hp,
                    'Y':hp,'V':hp
                  }
    groupings = {}
    for res in ordered_contacts:
        groupings[res] = np.unique([AA_groups[x] for x in ordered_contacts[res]])

    return groupings

def calc_contact_sim(native_contacts, pred_contacts):
    """Calculate the contact similarity between the native and pred contacts
    The contacts here represent the residues interacting with each interface residue.
    """

    n_matches = 0
    n_tot = len(native_contacts.keys())
    for res in native_contacts:
        if res in pred_contacts.keys():
            #If there is any match - count
            if np.intersect1d(native_contacts[res],pred_contacts[res]).shape[0]>0:
                n_matches+=1
    return n_matches/n_tot




def calc_metrics(pred_name, native_receptor_CA_coords, binder_CA_coords,
                native_receptor_if_res, binder_if_res,
                COM, grouped_native_contacts):
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


    #Superpose the receptor CAs and compare the centre of mass
    sup = SVDSuperimposer()
    all_pred_coords, pred_CA_coords, pred_CB_coords, pred_resnos, pred_seqs, pred_plDDT  = read_pdb(pred_name)
    #Superpose the CA coords of the receptor
    sup.set(native_receptor_CA_coords, pred_CA_coords['A']) #(reference_coords, coords)
    sup.run()
    rot, tran = sup.get_rotran()
    #Rotate the design coords to match the centre of mass for the native comparison
    rotated_coords = np.dot(pred_CA_coords['B'], rot) + tran
    rotated_CM =  np.sum(rotated_coords,axis=0)/(rotated_coords.shape[0])
    delta_CM = np.sqrt(np.sum(np.square(COM-rotated_CM)))
    #Rotate the pred CBs of the receptor
    pred_receptor_CBs = pred_CB_coords['A']
    pred_receptor_CBs = np.dot(pred_receptor_CBs, rot) + tran
    #Calculate the interface RMSD
    #binder_rmsd = calc_if_rmsd(binder_CA_coords[binder_if_res], rotated_coords[binder_if_res])
    #receptor_rmsd = calc_if_rmsd(native_receptor_CB_coords[native_receptor_if_res], pred_receptor_CBs[native_receptor_if_res])

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

    #Get the residue groupings for the predicted receptor contacts
    # mat = np.append(pred_CB_coords['A'], pred_CB_coords['B'],axis=0)
    # a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
    # dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    # l1 = len(pred_CB_coords['A'])
    # #Get interface
    # contact_dists = dists[:l1,l1:] #first dimension = receptor, second = peptide
    # pred_contacts = np.argwhere(contact_dists<8)
    # #Group the contacts
    # grouped_pred_contacts = group_residues(pred_contacts, pred_binder_seq)
    #
    # #Calculate the contact similarity btw the design and native binders
    # frac_rec_contacts = calc_contact_sim(grouped_native_contacts, grouped_pred_contacts)

    return (closest_dists_binder, closest_dists_receptor, pred_plDDT, delta_CM)





#################MAIN####################
def run_scoring(native_structure, seed_df, row_num, preds):
    """Run function for the scoring
    """

    all_native_coords, native_CA_coords, native_CB_coords, native_resnos, native_seqs, native_plDDT = read_pdb(native_structure)


    #The target receptor chain is always A and the binder B here
    receptor_chain, binder_chain = 'A', 'B'
    #Get native sequences
    native_receptor_seq = native_seqs[receptor_chain]
    native_binder_seq = native_seqs[binder_chain]
    #Get receptor if res and binder COM
    (receptor_if_res, binder_if_res, COM, receptor_CAs,
    binder_CAs, receptor_binder_contacts) = get_native_features(native_CA_coords, native_CB_coords,
                                                                receptor_chain, binder_chain, native_binder_seq)


    #Go through all crops
    results = {'if_dist_binder':[], 'if_dist_receptor':[], 'plddt':[], 'delta_CM':[] }

    #Get the crop region
    row = seed_df.loc[row_num]
    binder_if_res_crop =  np.argwhere((binder_if_res>=row.cs)&(binder_if_res<row.ce))[:,0]
    binder_if_res_crop-=binder_if_res_crop[0]

    #Get the contacts for the crop
    crop_contacts = np.argwhere((receptor_binder_contacts[:,1]>=row.cs)&(receptor_binder_contacts[:,1]<row.ce))[:,0]
    crop_contacts = receptor_binder_contacts[crop_contacts]
    #Group them
    grouped_native_contacts = group_residues(crop_contacts, native_binder_seq)


    #Get all the predicted files
    n_preds = len(preds)
    #Go through all preds
    for i in range(n_preds):

        #Get the loss metrics
        (closest_dists_binder, closest_dists_receptor,
        pred_plDDT, delta_CM) = calc_metrics(pred_dir+'unrelaxed_'+target_id+'_'+str(i+1)+'.pdb',
                                        receptor_CAs, binder_CAs[row.cs:row.ce],
                                        receptor_if_res, binder_if_res_crop,
                                        COM, grouped_native_contacts)


        #Get the seq recovery in the binder interface pos
        #binder_if_seq = ''.join(pred_binder_seq[binder_if_res_crop])
        #native_if_seq = ''.join(native_binder_seq[binder_if_res_crop])
        #binder_if_seq_rec = np.mean([(a==b) for a, b in zip(binder_if_seq, native_if_seq)])

        #Save
        results['if_dist_binder'].append(closest_dists_binder.mean())
        results['if_dist_receptor'].append(closest_dists_receptor.mean())
        results['plddt'].append(pred_plDDT['B'].mean())
        results['delta_CM'].append(delta_CM )
        #results['binder_if_CA_rmsd'].append(binder_rmsd)
        #results['receptor_if_CB_rmsd'].append(receptor_rmsd)
        #results['receptor_if_seq_recovery'].append(if_recovery)
        #results['receptor_overall_seq_recovery'].append(overall_recovery)
        #results['frac_recovered_contacts'].append(frac_rec_contacts)
        #results['binder_if_seq_rec'].append(binder_if_seq_rec)

        print(i)
    #Df
    results_df = pd.DataFrame.from_dict(results)
    return results_df
