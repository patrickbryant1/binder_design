import pandas as pd
import glob
import numpy as np
import pdb

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


aln_seqs, pdb_ids, pdb_chains = parse_results('../../data/Foldseek_results/')
write_ids_for_download(pdb_ids, '../../data/Foldseek_results/mmcif/ids.txt')
