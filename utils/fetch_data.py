import pandas as pd
from sklearn.datasets.base import Bunch

def fetch_data(data_dir, file_suffix, metadata='metadata', subject_group=None):
    """Get data corresponding to the IDs in the metadata file.

    Parameters
    ----------
    data_dir: string
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

   subjects: list
        List of subject IDs to retrieve from data_dir. Used to select specific
        subjects and for reproducibility. 

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interesting attributes are :
         - 'func': Paths to functional ASL images
         - 'metadata': Pandas dataframe of study-specific relevant data
    """

    # First, get the metadata
    # Load the csv file
    metadata = pd.read_csv(data_dir + metadata + '.csv')

    if 'Imaging_Y_N' in metadata.columns:
        # Select only particpants with asl images
        metadata = metadata[metadata['Imaging_Y_N'] == 1]

    # Keep metadata for selected subjects
    if subject_group is not None:
        metadata = metadata[metadata['Diagnosis'].isin(subject_group)]

    subjects = metadata['SDYID']
    print(len(subjects), "<<<<<< CHECK IT")
    print(metadata)
    #print('Loading example sub:', data_dir+'sub-%s/ses-pre/asl/sub-%s_%s'%(subjects[0],subjects[0],file_suffix))

    # Fetch filenames
    images = [
        data_dir + 'sub-%s/ses-pre/asl/sub-%s_%s'%(i,i,file_suffix)
#        data_dir + '%s/%s'%(i,file_suffix)
        #data_dir + 'struc/struc/T1_%s_%s'%(i,file_suffix)
        for i in subjects
    ]

#    struc_imgs = [
#        data_dir + 'struc/struc/T1_%s_struc_struc_GM_to_template_GM_thr.nii.gz'%i
#        for i in subjects
#    ]

    return Bunch(imgs=images, 
#                 struc_imgs=struc_imgs,
                 metadata=metadata, description='CnC_ASL')
