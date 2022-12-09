#Functions for making an alignment of structural alignments from Foldseek
import pandas as pd
import numpy as np
import glob
import pdb


def parse_results(results_dir, db='pdb100'):
    """Parse the foldseek results
    """

    df = pd.read_csv(glob.glob(results_dir+'*_'+db+'.m8')[0], sep='\t', header=None)

    aln_seqs = df[15].values
    pdb_ids = [x.split('_')[0] for x in df[1].values]
    pdb_chains = [x.split('_')[1] for x in df[1].values]

    return aln_seqs, pdb_ids, pdb_chains



def make_aln(query_seq, results_dir, eval_t=0.001):
    """Make a structural alignment from all hits
    """

    hits = glob.glob(results_dir+'*.m8')
    aln = [query_seq]
    query_len = len(query_seq)
    for hitname in hits:
        df = pd.read_csv(hitname, sep='\t', header=None)
        #Column 10 is the eval
        df = df[df[10]<eval_t]
        if len(df)<1:
            continue
        #Add the hit to the alignment - column 15
        #Add according to: 6,7; qstart,qend. qstart is the start pos
        qstart, qend, tstart, tend, hits = df[6].values, df[7].values, df[8].values, df[9].values, df[15].values
        for i in range(len(qstart)):
            qs_i = qstart[i]-1
            remainder = query_len-qend[i]
            hit_aln = hits[i][tstart[i]-1:tend[i]]
            aln_seq = '-'*qs_i+hit_aln+'-'*remainder
            if len(aln_seq)!=query_len:
                print('mismatch!')
                pdb.set_trace()
            else:
                aln.append(aln_seq)

        pdb.set_trace()




query_seq = 'PQFTAGNSHVAQNRRNYMDPSYKLEKLRDIPEEDIVRLLAHRAPGEEYKSIHPPLEEMEEPDCAVRQIVKPTEGAAAGDRIRYVQYTDSMFFSPITPYQRAWEALNRYKGVDPGVLSGRTIIEARERDIEKIAKIEVDCELYDTARTGLRGRTVHGHAVRLDKDGMMFDALRRWSRGADGTVTYVKDMIGGAMDKEVTLGKPLSDAELLKKTTMYRNAQGGVWQEADDPESMDVTAQIHWKRSVGGFQPWAKMKDIKGGKKDVGVKNLKLFTPRGGVE'
make_aln(query_seq, '/home/patrick/binder_design/data/Foldseek_results/')
