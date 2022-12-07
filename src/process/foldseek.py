import pandas as pd
import glob
import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from collections import Counter
import pdb


##########Parse Foldseek results###########
def parse_results(results_dir, db='pdb100'):
    """Parse the foldseek results
    """

    df = pd.read_csv(glob.glob(results_dir+'*_'+db+'.m8')[0], sep='\t', header=None)

    aln_seqs = df[15].values
    pdb_ids = [x.split('_')[0] for x in df[1].values]
    pdb_chains = [x.split('_')[1] for x in df[1].values]

    return aln_seqs, pdb_ids, pdb_chains

def write_ids_for_download(pdb_ids, outname):
    """Write the unique pdb ids to a list for download
    """

    with open(outname, 'w') as file:
        for id in np.unique(pdb_ids):
            file.write(id+',')


###########Read hits and select seeds##########
def read_pdb(pdbname):
    '''Read PDB
    '''

    three_to_one = {'ARG':'R', 'HIS':'H', 'LYS':'K', 'ASP':'D', 'GLU':'E', 'SER':'S', 'THR':'T', 'ASN':'N', 'GLN':'Q', 'CYS':'C', 'GLY':'G', 'PRO':'P', 'ALA':'A', 'ILE':'I', 'LEU':'L', 'MET':'M', 'PHE':'F', 'TRP':'W', 'TYR':'Y', 'VAL':'V',
    'SEC':'U', 'PYL':'O', 'GLX':'X', 'UNK': 'X'}

    if pdbname[-3:]=='pdb':
        parser = PDBParser()
    else:
        parser = MMCIFParser()
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
                    model_seqs[chain.id].append(three_to_one[res_name])
                    model_atoms[chain.id].append(atom_id)
                    model_resnos[chain.id].append(residue.get_id()[1])



    #Convert to array
    for key in model_coords:
        model_coords[key] = np.array(model_coords[key])
        model_seqs[key] = np.array(model_seqs[key])
        model_atoms[key] = np.array(model_atoms[key])
        model_resnos[key] = np.array(model_resnos[key])

    return model_coords, model_seqs, model_atoms, model_resnos

def get_intres(coords1, coords2, resnos1, resnos2, atoms1, atoms2):
    '''Get the protein-NA interactions
    '''

    #Get CBs
    inds1 = np.argwhere(atoms1=='CB')[:,0]
    inds2 = np.argwhere(atoms2=='CB')[:,0]
    coords1, coords2 = coords1[inds1], coords2[inds2]
    resnos1, resnos2 = resnos1[inds1], resnos2[inds2]

    #Calc 2-norm - distance between chains
    mat = np.append(coords1, coords2, axis=0)
    a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    #Get interface
    l1 = len(coords1)
    contact_dists = dists[:l1,l1:] #first dimension = ch1, second = ch2
    contacts = np.argwhere(contact_dists<8)
    if len(contacts)>0:
        #Get residue numbers
        intres2 = resnos2[contacts[:,1]]

        return intres2
    else:
        return []

def crop(intres, resnos, croplens):
    """Crop structure by selecting the highest density of contacts
    """

    #Get the unique resnos
    u_res = np.unique(resnos)
    res_counts = np.zeros((u_res.shape[0]))
    counts = Counter(intres)
    #Assign counts
    for key in counts:
        res_counts[np.argwhere(u_res==key)[0][0]]=counts[key]

    #Go through the res_counts and crop
    selected_crops = {'contacts':[], 'cs':[], 'ce':[]}
    for croplen in croplens:
        #Add new
        selected_crops['contacts'].append(0)
        selected_crops['cs'].append(0)
        selected_crops['ce'].append(0)
        for i in range(len(res_counts)-croplen):
            #Add the crop if more contacts
            n_contacts = np.sum(res_counts[i:i+croplen])
            if n_contacts>selected_crops['contacts'][-1]:
                #Remove prev
                selected_crops['contacts'].pop()
                selected_crops['cs'].pop()
                selected_crops['ce'].pop()
                #Add new
                selected_crops['contacts'].append(n_contacts)
                selected_crops['cs'].append(i)
                selected_crops['ce'].append(i+croplen)

    #Df
    #The positions represent the unique residue positions
    crop_df = pd.DataFrame.from_dict(selected_crops)
    crop_df['croplen']=crop_df.ce-crop_df.cs
    return crop_df

def get_interaction_seeds(pdb_ids, pdb_chains, mmcifdir, min_len, max_len, outdir):
    """Parse the hits and extract interaction seeds
    """

    croplens = np.arange(min_len,max_len+1)
    possible_seeds = {'PDB_ID':[], 'target_chain':[], 'seed_chain':[],
                        'contacts':[], 'cs':[], 'ce':[], 'croplen':[]}
    for i in range(len(pdb_ids)):
        #Read the model
        model_coords, model_seqs, model_atoms, model_resnos = read_pdb(mmcifdir+pdb_ids[i]+'.cif')
        #Go through all chains
        target_chain = pdb_chains[i]
        seed_chains = np.setdiff1d([*model_coords.keys()], target_chain)
        for seed_chain in seed_chains:
            #Get the interacting residues
            #Intres 2 are all interactions with the target chain from the seed chain
            intres2 = get_intres(model_coords[target_chain], model_coords[seed_chain],
                                        model_resnos[target_chain], model_resnos[seed_chain],
                                        model_atoms[target_chain], model_atoms[seed_chain])
            #Check
            if len(intres2)>0:
                crop_df = crop(intres2, model_resnos[seed_chain], croplens)
                #Get the best crop
                best_crops = crop_df[crop_df.contacts==crop_df.contacts.max()]
                best_crop = best_crops[best_crops.croplen==best_crops.croplen.min()]
                #Save
                possible_seeds['PDB_ID'].append(pdb_ids[i])
                possible_seeds['target_chain'].append(target_chain)
                possible_seeds['seed_chain'].append(seed_chain)
                possible_seeds['contacts'].append(best_crop.contacts.values[0])
                possible_seeds['cs'].append(best_crop.cs.values[0])
                possible_seeds['ce'].append(best_crop.ce.values[0])
                possible_seeds['croplen'].append(best_crop.croplen.values[0])

    #Create df
    seed_df = pd.DataFrame.from_dict(possible_seeds)
    #Save
    seed_df.to_csv(outdir+'seed_df.csv', index=None)
    return seed_df


def calc_COM():
    """Calculate the COM for a seed after aligning the CAs
    of the search structure and the hit chains
    """

def write_seeds_for_design(seed_df, search_structure, min_contacts_per_pos=1):
    """Write seeds that differ in COM towards the target more than X Å
    """
    #Select seeds
    seed_df['contacts_per_pos'] = seed_df.contacts/seed_df.croplen
    seed_df = seed_df.sort_values(by='contacts_per_pos', ascending=False)
    seed_df = seed_df[seed_df.contacts_per_pos>=min_contacts_per_pos]

    #Calculate the COM for all seeds
    search_coords, search_seqs, search_atoms, search_resnos = read_pdb(search_structure)
    search_chain = [*search_coords.keys()][0]
    #Get the CA coords
    search_CA_coords = search_coords[search_chain][np.argwhere(search_atoms[search_chain]=='CA')[:,0]]
    search_seq = ''.join(search_seqs[search_chain][np.argwhere(search_atoms[search_chain]=='CA')[:,0]])
    pdb.set_trace()

############################MAIN#############################
#Process
aln_seqs, pdb_ids, pdb_chains = parse_results('../../data/Foldseek_results/')
#write_ids_for_download(pdb_ids, '../../data/Foldseek_results/mmcif/ids.txt')
#Get seeds
mmcfdir = '../../data/Foldseek_results/mmcif/'
outdir = '../../data/Foldseek_results/mmcif/seeds/'
min_len, max_len = 10, 50
try:
    seed_df = pd.read_csv(outdir+'seed_df.csv')
except:
    pdb.set_trace()
    seed_df = get_interaction_seeds(pdb_ids, pdb_chains, mmcfdir, min_len, max_len, outdir)
#Pick seeds based on contact density and COM diff (avoid repetitive seeds)
write_seeds_for_design(seed_df, '../../data/3SQG_C.pdb', min_contacts_per_pos=1)
