# Module to compute measures of covariance.

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from nilearn import connectome

def correlation(data, kind='pearson'):
    ''' Compute correlation matrix of a series of voxels. Options other than
    Pearson's r to be implemented in the future.

    Args:
        data    numpy array. Variables to be correlated are given by the rows.

    Returns: 
        Numpy array of size (data.shape[0], data.shape[0])
    '''

    print('Computing sample correlation with %s estimator'%kind)

    # Compute correlation matrix of rows of 'data'.
    if kind is 'spearman':
        cor_matrix, p_vals = stats.spearmanr(data.T)
    if kind is 'ledoit':
        cor_matrix = connectome.ConnectivityMeasure(kind='correlation').fit_transform([data.T,])[0]
    elif kind is 'pearson':
        cor_matrix = np.corrcoef(data)
        cor_matrix[np.isnan(cor_matrix)] = 0.

    if kind is not 'spearman':
        # Compute p-value of correlation using two-sided t-test with data.shape[1]-2.
        # degrees of freedom. 
        cor_matrix[cor_matrix > 0.999] = 0.999999
        t_stats = cor_matrix * np.sqrt(data.shape[1] - 2.)\
                  / np.sqrt(1. - cor_matrix**2.)
        p_vals = stats.t.sf(np.abs(t_stats), data.shape[1]-2.)*2.



    return cor_matrix, p_vals

def get_components(data, n_components, mode='PCA'):
    ''' Compute PCA or ICA of matrix. Methods leveraged from scikit-learn.
    '''

    transformer = PCA(n_components=n_components) 

    transformer.fit(data)

    return transformer

def parcellate_and_correlate(niftis_list, output_dir, parcellator, prefix=None, detrend=False, pve_gm_imgs=None):
    ''' Compute parcellation and covariance of a number of nifti images.
    '''

    cov_pvals = list()
    compressed_data = list()
    compressed_strucs = list()
    for niftis in niftis_list:    

        # Reduce data into parcels
        reduced_data = parcellator.transform(niftis)

        compressed_data.append(reduced_data)
        if detrend:
            image_means = np.mean(reduced_data, axis=1, keepdims=True)
            reduced_data -= image_means

        #reduced_data = np.nan_to_num(reduced_data)
        #reduced_data *= 1./reduced_struc

        # Calculate correlations and p-values. Puts 0.999999 on the diagonal.
        covariance, pvals = correlation(reduced_data.T, kind='pearson')    

        cov_pvals.append([covariance, pvals])

    for name, number in zip(['hc', 'bd'], [0, 1]):
        np.save(output_dir + prefix + 'reduced_%s'%name, compressed_data[number])
#        np.save(output_dir + prefix + 'reduced_%s_strucs'%name, compressed_strucs[number])

        #compressed_brain[number].to_filename(output_dir+prefix+'compressed_%s.nii.gz'%name)

    return cov_pvals