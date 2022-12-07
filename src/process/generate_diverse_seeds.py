import pandas as pd
import glob
import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from collections import Counter
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio import pairwise2
import pdb


##########Parse Foldseek results###########
def write_input_pdb(chain_coords, chain_seq, chain_atoms, chain_resnos, chain, outname):
    """Write a PDB file of the seeds
    """

    one_to_three = {'R':'ARG', 'H':'HIS', 'K':'LYS', 'D':'ASP', 'E':'GLU', 'S':'SER', 'T':'THR',
                    'N':'ASN', 'Q':'GLN', 'C':'CYS', 'G':'GLY', 'P':'PRO', 'A':'ALA', 'I':'ILE',
                    'L':'LEU', 'M':'MET', 'F':'PHE', 'W':'TRP', 'Y':'TYR', 'V':'VAL', 'U':'SEC',
                    'O':'PYL', 'X':'GLX', 'X':'UNK'}

    with open(outname, 'w') as file:
        #Write chain 1
        atmno=1
        for i in range(len(chain_coords)):
            x,y,z = chain_coords[i]
            x,y,z = str(np.round(x,3)), str(np.round(y,3)), str(np.round(z,3))
            file.write(format_line(str(atmno), chain_atoms[i], one_to_three[chain_seq[i]],
            chain, str(chain_resnos[i]), x,y,z, '1.00','100', chain_atoms[i][0])+'\n')
            atmno+=1


def prepare_input(pdbfile, chain, outdir, pdbid):
    """Parse the PDB file and write the intended chain
    """

    #Read
    model_coords, model_seqs, model_atoms, model_resnos = read_pdb(pdbfile)

    #Write the selected chain
    write_input_pdb(model_coords[chain], model_seqs[chain], model_atoms[chain], model_resnos[chain], chain, outdir+pdbid+'_'+chain+'.pdb')



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


def calc_COM(search_CA_coords, search_seq,
        target_N_coords, target_CA_coords, target_C_coords, target_seq,
        seed_N_coords, seed_CA_coords, seed_C_coords):
    """Calculate the COM for a seed after aligning the CAs
    of the search structure and the hit chains
    """

    #Align the search and target seqs to get matching CA regions
    alignment = pairwise2.align.globalxx(search_seq, target_seq)[0]
    search_aln, target_aln = alignment[0], alignment[1]
    #Go through the alignment and get the matching pos
    keep_search, keep_target = [], []
    si, ti = 0, 0 #Keep track of the indices
    for i in range(len(search_aln)):

        if (search_aln[i]!='-') and (target_aln[i]!='-'):
            keep_search.append(si)
            keep_target.append(ti)
        if search_aln[i]!='-':
            si+=1
        if target_aln[i]!='-':
            ti+=1

    #Structural superposition of the search and target (hit) CA coords
    sup = SVDSuperimposer()

    sup.set(search_CA_coords[keep_search], target_CA_coords[keep_target]) #(reference_coords, coords)
    sup.run()
    rot, tran = sup.get_rotran()
    #Rotate the coords to match for the centre of mass calc
    rotated_target_N_coords = np.dot(target_N_coords, rot) + tran
    rotated_target_CA_coords = np.dot(target_CA_coords, rot) + tran
    rotated_target_C_coords = np.dot(target_C_coords, rot) + tran
    rotated_seed_N_coords = np.dot(seed_N_coords, rot) + tran
    rotated_seed_CA_coords = np.dot(seed_CA_coords, rot) + tran
    rotated_seed_C_coords = np.dot(seed_C_coords, rot) + tran
    rotated_CM =  np.sum(rotated_seed_CA_coords,axis=0)/(rotated_seed_CA_coords.shape[0])
    #Cat
    rotated_target_coords = np.concatenate([np.expand_dims(rotated_target_N_coords,axis=1), np.expand_dims(rotated_target_CA_coords,axis=1), np.expand_dims(rotated_target_C_coords,axis=1)],axis=1)
    rotated_seed_coords = np.concatenate([np.expand_dims(rotated_seed_N_coords,axis=1), np.expand_dims(rotated_seed_CA_coords,axis=1), np.expand_dims(rotated_seed_C_coords,axis=1)],axis=1)

    return rotated_target_coords, rotated_seed_coords, rotated_CM

def format_line(atm_no, atm_name, res_name, chain, res_no, x,y,z,occ,B,atm_id):
    '''Format the line into PDB
    '''

    #Get blanks
    atm_no = ' '*(5-len(atm_no))+atm_no
    atm_name = atm_name+' '*(4-len(atm_name))
    res_no = ' '*(4-len(res_no))+res_no
    x =' '*(8-len(x))+x
    y =' '*(8-len(y))+y
    z =' '*(8-len(z))+z
    occ = ' '*(6-len(occ))+occ
    B = ' '*(6-len(B))+B

    line = 'ATOM  '+atm_no+'  '+atm_name+res_name+' '+chain+res_no+' '*4+x+y+z+occ+B+' '*11+atm_id+'  '
    return line

def write_pdb(target_coords, target_seq, seed_coords, seed_seq, outname):
    """Write a PDB file of the seeds
    """

    one_to_three = {'R':'ARG', 'H':'HIS', 'K':'LYS', 'D':'ASP', 'E':'GLU', 'S':'SER', 'T':'THR',
                    'N':'ASN', 'Q':'GLN', 'C':'CYS', 'G':'GLY', 'P':'PRO', 'A':'ALA', 'I':'ILE',
                    'L':'LEU', 'M':'MET', 'F':'PHE', 'W':'TRP', 'Y':'TYR', 'V':'VAL', 'U':'SEC',
                    'O':'PYL', 'X':'GLX', 'X':'UNK'}

    with open(outname, 'w') as file:
        atmno, resno = 1, 1
        #Write chain 1
        chain='A'
        for i in range(len(target_coords)):
            ai=0
            for atom in ['N', 'CA', 'C']:
                x,y,z = target_coords[i,ai]
                x,y,z = str(np.round(x,3)), str(np.round(y,3)), str(np.round(z,3))
                file.write(format_line(str(atmno), atom, one_to_three[target_seq[resno-1]],
                chain, str(resno), x,y,z, '1.00','100',atom[0])+'\n')
                atmno+=1
                ai+=1 #Atom index

            resno+=1
        #Write chain 2
        chain='B'
        resno=1
        for i in range(len(seed_coords)):
            ai=0
            for atom in ['N', 'CA', 'C']:
                x,y,z = seed_coords[i,ai]
                x,y,z = str(np.round(x,3)), str(np.round(y,3)), str(np.round(z,3))
                file.write(format_line(str(atmno), atom, one_to_three[seed_seq[resno-1]],
                chain, str(resno), x, y, z, '1.00','100',atom[0])+'\n')
                atmno+=1
                ai+=1 #Atom index



def write_seeds_for_design(seed_df, search_structure, mmcifdir, outdir, min_contacts_per_pos=1, COM_min_dist=2):
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

    #Go through the possible seeds and calculate their COM
    targets, t_seqs, seeds, s_seqs, COMs = [], [], [], [], []
    for ind, row in seed_df.iterrows():
        #Read the chains
        seed_coords, seed_seqs, seed_atoms, seed_resnos = read_pdb(mmcifdir+row.PDB_ID+'.cif')
        #Target
        target_N_coords = seed_coords[row.target_chain][np.argwhere(seed_atoms[row.target_chain]=='N')[:,0]]
        target_CA_coords = seed_coords[row.target_chain][np.argwhere(seed_atoms[row.target_chain]=='CA')[:,0]]
        target_C_coords = seed_coords[row.target_chain][np.argwhere(seed_atoms[row.target_chain]=='C')[:,0]]
        target_seq = ''.join(seed_seqs[row.target_chain][np.argwhere(seed_atoms[row.target_chain]=='CA')[:,0]])
        #Seed
        seed_N_coords = seed_coords[row.seed_chain][np.argwhere(seed_atoms[row.seed_chain]=='N')[:,0]][row.cs:row.ce+1]
        seed_CA_coords = seed_coords[row.seed_chain][np.argwhere(seed_atoms[row.seed_chain]=='CA')[:,0]][row.cs:row.ce+1]
        seed_C_coords = seed_coords[row.seed_chain][np.argwhere(seed_atoms[row.seed_chain]=='C')[:,0]][row.cs:row.ce+1]
        seed_seq = ''.join(seed_seqs[row.seed_chain][np.argwhere(seed_atoms[row.seed_chain]=='CA')[:,0]])[row.cs:row.ce+1]
        #Get the COM
        rotated_target_coords, rotated_seed_coords, rotated_CM = calc_COM(search_CA_coords, search_seq,
                                                                        target_N_coords, target_CA_coords, target_C_coords, target_seq,
                                                                        seed_N_coords, seed_CA_coords, seed_C_coords)
        #Save
        targets.append(rotated_target_coords)
        t_seqs.append(target_seq)
        seeds.append(rotated_seed_coords)
        s_seqs.append(seed_seq)
        COMs.append(rotated_CM)

    #Add COM to dfrow.PDB_ID
    seed_df['COM']=COMs
    seed_df['target_coords']=targets
    seed_df['target_seq']=t_seqs
    seed_df['seed_cords']=seeds
    seed_df['seed_seq']=s_seqs
    #Pick seeds with highest contact densities and delta COM >2
    seed_no = 1
    for ind, row in seed_df.iterrows():
        if ind>0:
            #Check COMs
            prev_COMs = seed_df.COM.values[:ind]
            #Get diff
            delta_COM = np.array([x for x in prev_COMs])-row.COM
            delta_COM = np.sqrt(np.sum(delta_COM**2,axis=1))
            if np.argwhere(delta_COM<=COM_min_dist).shape[0]>0:
                print(min(delta_COM))
                continue

        #Write the seed
        write_pdb(row.target_coords, row.target_seq, row.seed_cords, row.seed_seq, outdir+'seed_'+str(seed_no)+'.pdb')
        seed_no+=1
############################MAIN#############################
#Process
#aln_seqs, pdb_ids, pdb_chains = parse_results('../../data/Foldseek_results/')
#write_ids_for_download(pdb_ids, '../../data/Foldseek_results/mmcif/ids.txt')
#Get seeds
#mmcifdir = '../../data/Foldseek_results/mmcif/'
#outdir = '../../data/Foldseek_results/mmcif/seeds/'
#min_len, max_len = 10, 50
#try:
#     seed_df = pd.read_csv(outdir+'seed_df.csv')
# except:
#     seed_df = get_interaction_seeds(pdb_ids, pdb_chains, mmcifdir, min_len, max_len, outdir)
# #Pick seeds based on contact density and COM diff (avoid repetitive seeds)
# write_seeds_for_design(seed_df, '../../data/3SQG_C.pdb', mmcifdir, outdir, min_contacts_per_pos=1, COM_min_dist=2)
