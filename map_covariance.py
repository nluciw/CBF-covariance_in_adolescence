# This script loads images and computes the covariance map for the sample.
# We assume all preprocessing except region extraction is completed (i.e. any
# normalization etc.).

import argparse
import os
import sys
from utils import covariance, statistics
import numpy as np
from utils.fetch_data import fetch_data
from nilearn.image import concat_imgs
from nilearn.input_data import NiftiLabelsMasker

def get_args():
    ''' Read the command-line arguments.
    '''

    parser = argparse.ArgumentParser(
                description='Analyze functional covariance.',\
                fromfile_prefix_chars='@')
    parser.add_argument('--nifti_name', help='name of file of nifti volumes')
    parser.add_argument('--output_dir', help='name of output directory')
    parser.add_argument('--output_prefix', help='name of prefix of output files')
    parser.add_argument('--atlas', help='name of seed region, as given in cache')
    parser.add_argument('--mask', help='name of mask')
    parser.add_argument('--metadata', help='group membership metadata')

    return parser.parse_args()

def main(args):

    # Set analysis directories
    root = '/net/synapse/nt/users/bmacintosh_lab/nluciw/'
    # Root directory of data.
    data_dir = root + 'data/EnF/sourcedata/'
    # Output directory.
    output_dir = root + 'outputs/perf_covar/' + args.output_dir
    # Create output if does not exist.
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))

    # Save command line 
    with open(output_dir+'commandline_args.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    # Load 4d nifti objects for both groups
    bd_data = fetch_data(data_dir, args.nifti_name, metadata=args.metadata, 
            subject_group=('BD',))
    hc_data = fetch_data(data_dir, args.nifti_name, metadata=args.metadata,
            subject_group=('HC',))

    hc_vols = concat_imgs(hc_data.imgs)
    bd_vols = concat_imgs(bd_data.imgs)

    # Construct or load parcellator object using the atlas we specify on 
    # the command line.
    parcellator = NiftiLabelsMasker(labels_img=args.atlas,
                                    mask_img=data_dir+'masks/cbf_80p_aal_merge_mni.nii.gz',
                                    standardize=False,
                                    strategy='mean')
    parcellator.fit()

    # Do the parcellation and correlation for both groups.
    hc_covar, bd_covar =\
        covariance.parcellate_and_correlate([hc_vols,bd_vols], 
                                            output_dir, 
                                            parcellator,
                                            prefix = args.output_prefix,
                                            detrend=False#,
#                                            pve_gm_imgs=[concat_imgs(hc_data.struc_imgs),
#                                                         concat_imgs(bd_data.struc_imgs)]
                                            )

    print(len(bd_data.imgs), len(hc_data.imgs))
    difference = statistics.compute_difference(bd_covar[0], 
                                               hc_covar[0],
                                               len(bd_data.imgs),
                                               len(hc_data.imgs))

    cors = np.stack((bd_covar[0], hc_covar[0], difference))

    np.save(output_dir + args.output_prefix + 'cors', cors)

if __name__ == '__main__':

    args = get_args()

    print('\n\nPerforming covariance analysis on %s'%args.nifti_name)
    print('Saving results in %s \n'%args.output_dir)

    main(args)
