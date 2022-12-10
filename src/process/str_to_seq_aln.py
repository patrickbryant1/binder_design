#Functions for making an alignment of structural alignments from Foldseek
import pandas as pd
import numpy as np
import glob
import pdb


def write_a3m(outname, aln, ids, evals):
    """Write the alignment in a3m
    """

    #Sort by eval
    aln, ids, evals = np.array(aln), np.array(ids), np.array(evals)
    aln, ids, evals = aln[np.argsort(evals)], ids[np.argsort(evals)], evals[np.argsort(evals)]
    with open(outname, 'w') as file:
        for i in range(len(ids)):
            file.write('>'+ids[i]+'\n')
            file.write(aln[i]+'\n')



def make_aln(query_seq, results_dir, eval_t=0.001):
    """Make a structural alignment from all hits
    """

    hits = glob.glob(results_dir+'*.m8')
    aln, ids, evals = [query_seq], ['Target_sequence'], [-1]
    query_len = len(query_seq)
    for hitname in hits:
        try:
            df = pd.read_csv(hitname, sep='\t', header=None)
        except:
            continue
        #Column 10 is the eval
        df = df[df[10]<eval_t]
        if len(df)<1:
            continue
        #Add the hit to the alignment - column 15
        #Cols 14 and 15 are the pairwise aln btw the query and hit.
        #Both of these can contain gaps. If there are gaps in the query - these are insertions and should be removed
        #Add according to: 6,7; qstart,qend. qstart is the start pos
        hit_ids, qstart, qend, tstart, tend, hit_evals, query_alns, hit_alns =df[1].values, df[6].values, df[7].values, df[8].values, df[9].values, df[10].values, df[14].values, df[15].values
        for i in range(len(qstart)):
            qs_i, qe_i, qaln = qstart[i]-1, qend[i], query_alns[i]
            if '-' in qaln:
                #Get the insertions and remove them from the hit aln
                insertions = np.argwhere(np.array([x for x in qaln])=='-')[:,0]
                keep_pos = np.setdiff1d(np.arange(len(qaln)),insertions)
                hit_aln = ''.join(np.array([x for x in hit_alns[i]])[keep_pos])
            else:
                hit_aln = hit_alns[i]

            remainder = query_len-qe_i
            aln_seq = '-'*qs_i+hit_aln+'-'*remainder
            if len(aln_seq)!=query_len:
                print('mismatch!')
                pdb.set_trace()
            else:
                aln.append(aln_seq)
                ids.append(hit_ids[i])
                evals.append(hit_evals[i])

    return aln, ids, evals





# query_seq = 'PQFTAGNSHVAQNRRNYMDPSYKLEKLRDIPEEDIVRLLAHRAPGEEYKSIHPPLEEMEEPDCAVRQIVKPTEGAAAGDRIRYVQYTDSMFFSPITPYQRAWEALNRYKGVDPGVLSGRTIIEARERDIEKIAKIEVDCELYDTARTGLRGRTVHGHAVRLDKDGMMFDALRRWSRGADGTVTYVKDMIGGAMDKEVTLGKPLSDAELLKKTTMYRNAQGGVWQEADDPESMDVTAQIHWKRSVGGFQPWAKMKDIKGGKKDVGVKNLKLFTPRGGVE'
# eval_t=1
# aln, ids, evals = make_aln(query_seq, '/home/patrick/binder_design/data/Foldseek_results/',eval_t)
# write_a3m( '/home/patrick/binder_design/data/Foldseek_results/hits.a3m', aln, ids, evals)
