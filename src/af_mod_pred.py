# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script.
Modified by Patrick Bryant to include custom loss and feature reuse for speed-up.
"""
import json
import os
import warnings
import pathlib
import pickle
import random
import sys
import time
from typing import Dict, Optional

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import msaonly
from alphafold.data import foldonly
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
#from alphafold.relax import relax
import numpy as np
import jax
from jax import numpy as jnp
from jax import grad, value_and_grad
import copy
from collections import defaultdict
import glob
import pdb
# Internal import (7716).

##### flags #####

############FUNCTIONS###########
def update_features(feature_dict, peptide_sequence):
    '''Update the features used for the pred:
    #Sequence features
    'aatype', 'between_segment_residues', 'domain_name',
    'residue_index', 'seq_length', 'sequence',
    #MSA features
    'deletion_matrix_int', 'msa', 'num_alignments',
    #Template features
    'template_aatype', 'template_all_atom_masks',
    'template_all_atom_positions', 'template_domain_names',
    'template_sequence', 'template_sum_probs'
    '''
    #Save
    new_feature_dict = {}
    peptide_features = pipeline.make_sequence_features(sequence=peptide_sequence,
                                                    description='peptide', num_res=len(peptide_sequence))
    #Merge sequence features
    #aatype
    new_feature_dict['aatype']=np.concatenate((feature_dict['aatype'],peptide_features['aatype']))
    #between_segment_residues
    new_feature_dict['between_segment_residues']=np.concatenate((feature_dict['between_segment_residues'],peptide_features['between_segment_residues']))
    #domain_name
    new_feature_dict['domain_name'] = feature_dict['domain_name']
    #residue_index
    new_feature_dict['residue_index']=np.concatenate((feature_dict['residue_index'],peptide_features['residue_index']+feature_dict['residue_index'][-1]+201))
    #seq_length
    new_feature_dict['seq_length']=np.concatenate((feature_dict['seq_length']+peptide_features['seq_length'][0],
                                            peptide_features['seq_length']+feature_dict['seq_length'][0]))
    #sequence
    new_feature_dict['sequence']=np.array(feature_dict['sequence'][0]+peptide_features['sequence'][0], dtype='object')

    #Merge MSA features
    #deletion_matrix_int
    new_feature_dict['deletion_matrix_int']=np.concatenate((feature_dict['deletion_matrix_int'],
                                            np.zeros((feature_dict['deletion_matrix_int'].shape[0],len(peptide_sequence)))), axis=1)
    #msa
    peptide_msa = np.zeros((feature_dict['deletion_matrix_int'].shape[0],len(peptide_sequence)),dtype='int32')
    peptide_msa[:,:] = 21
    HHBLITS_AA_TO_ID = {'A': 0,'B': 2,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'J': 20,'K': 8,'L': 9,'M': 10,'N': 11,
                        'O': 20,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'U': 1,'V': 17,'W': 18,'X': 20,'Y': 19,'Z': 3,'-': 21,}
    for i in range(len(peptide_sequence)):
        peptide_msa[0,i]=HHBLITS_AA_TO_ID[peptide_sequence[i]]
    new_feature_dict['msa']=np.concatenate((feature_dict['msa'], peptide_msa), axis=1)
    #num_alignments
    new_feature_dict['num_alignments']=np.concatenate((feature_dict['num_alignments'], feature_dict['num_alignments'][:len(peptide_sequence)]))

    #Merge template features
    for key in ['template_aatype', 'template_all_atom_masks', 'template_all_atom_positions',
                'template_domain_names', 'template_sequence', 'template_sum_probs']:
        new_feature_dict[key]=feature_dict[key]

    return new_feature_dict


def predict_function(peptide_sequence, feature_dict, output_dir, model_runners, random_seed):
    '''Predict
    '''

    #Add features for the binder
    #Update features
    new_feature_dict = update_features(feature_dict, peptide_sequence)
    # Write out features as a pickled dictionary.
    features_output_path = os.path.join(output_dir, 'features.pkl')
    with open(features_output_path, 'wb') as f:
      pickle.dump(new_feature_dict, f, protocol=4)
    # Run the model.
    for model_name, model_runner in model_runners.items():
      logging.info('Running model %s', model_name)
      processed_feature_dict = model_runner.process_features(
          new_feature_dict, random_seed=random_seed)

      t_0 = time.time()
      prediction_result = model_runner.predict(processed_feature_dict)
      print('Prediction took', time.time() - t_0,'s')

    # Get the pLDDT confidence metric.
    plddt = prediction_result['plddt']
    #Get the protein
    plddt_b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(features=processed_feature_dict,result=prediction_result,b_factors=plddt_b_factors)

    return unrelaxed_protein

def parse_atm_record(line):
    '''Get the atm record
    '''
    record = defaultdict()
    record['name'] = line[0:6].strip()
    record['atm_no'] = int(line[6:11])
    record['atm_name'] = line[12:16].strip()
    record['atm_alt'] = line[17]
    record['res_name'] = line[17:20].strip()
    record['chain'] = line[21]
    record['res_no'] = int(line[22:26])
    record['insert'] = line[26].strip()
    record['resid'] = line[22:29]
    record['x'] = float(line[30:38])
    record['y'] = float(line[38:46])
    record['z'] = float(line[46:54])
    record['occ'] = float(line[54:60])
    record['B'] = float(line[60:66])

    return record

def save_design(unrelaxed_protein, output_dir, model_name, l1):
    '''Save the resulting protein-peptide design to a pdb file
    '''

    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    chain_name = 'A'
    with open(unrelaxed_pdb_path, 'w') as f:
        pdb_contents = protein.to_pdb(unrelaxed_protein).split('\n')
        for line in pdb_contents:
            try:
                record = parse_atm_record(line)
                if record['res_no']>l1:
                    chain_name='B'
                outline = line[:21]+chain_name+line[22:]
                f.write(outline+'\n')
            except:
                f.write(line+'\n')


def predict_binder(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    msas: str,
    data_pipeline: pipeline.DataPipeline,
    random_seed: int,
    model_runners: Optional[Dict[str, model.RunModel]],
    peptide_sequences: list,
    target_id: str):

  """
  1. Make all input features to AlphaFold: update_features
  3. Predict the structure: model_runner.predict(processed_feature_dict)
  """
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Get features. This only applies to the receptor
  feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        input_msas=msas,
        template_search=None,
        msa_output_dir=None)

  #Check for previous designs
  num_finished_designs = len(glob.glob(output_dir+'unrelaxed_'+target_id+'*.pdb'))
  #batch
  peptide_sequences = np.array(peptide_sequences)[num_finished_designs:]
  #Iterate
  n_preds = num_finished_designs

  #Go through the seqs
  for peptide_sequence in peptide_sequences:
    #Predict
    unrelaxed_protein = predict_function(peptide_sequence, feature_dict, output_dir, model_runners, random_seed)
    n_preds+=1
    #Save the designs
    save_design(unrelaxed_protein, output_dir_base, target_id+'_'+str(n_preds), feature_dict['seq_length'][0])

######################MAIN###########################

def run_preds(receptor_fasta_path, msa, num_ensemble, max_recycles, data_dir, peptide_sequences, output_dir, target_id):
    """Run the predictions
    """

    #Use a single ensemble
    num_ensemble = 1

    # Check for duplicate FASTA file names.
    fasta_name = pathlib.Path(receptor_fasta_path).stem

    data_pipeline = foldonly.FoldDataPipeline()
    model_runners = {}
    model_config = config.model_config('model_1')
    model_config.data.eval.num_ensemble = num_ensemble
    model_config.data.common.num_recycle = max_recycles
    model_config.model.num_recycle = max_recycles
    model_params = data.get_model_haiku_params(model_name=model_name, data_dir=data_dir)
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner

    logging.info('Have %d models: %s', len(model_runners),
                 list(model_runners.keys()))
    amber_relaxer = None
    #Seed
    random_seed = random.randrange(sys.maxsize)
    logging.info('Using random seed %d for the data pipeline', random_seed)

    # Predict structure for each of the sequences.
    predict_binder(fasta_path=receptor_fasta_path,
                    fasta_name=fasta_name,
                    output_dir_base=output_dir,
                    msas=[msa],
                    data_pipeline=data_pipeline,
                    random_seed=random_seed,
                    model_runners=model_runners,
                    peptide_sequences=peptide_sequences,
                    target_id=target_id)
