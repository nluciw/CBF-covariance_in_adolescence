# Compute statistics on test stats or p-values.

import numpy as np
from scipy import ndimage

class Correct:
    ''' Correct for multiple comparisons. To include: Bonferroni, FDR, SGoF, 
    permutation. Should probably make these class methods...
    '''
    def cluster_threshold(self, p_vals, p_threshold=0.001, cluster_size=1):
        ''' Returns true where the voxel passes the two-level thresholding.
        '''

        p_thresh = np.ma.masked_greater(p_vals, p_threshold)
        p_thresh = np.ma.masked_equal(p_thresh, 0.)

        clusters, num_clust = ndimage.measurements.label(np.invert(p_thresh.mask))

        sub = 0
        for i in range(1, num_clust+1):
            if np.where(clusters==i)[0].shape[0]<cluster_size:
                clusters[clusters==i] = 0
                sub += 1

        print('%03d clusters were found during two-level thresholding'%num_clust-sub)

        p_thresh[clusters < 1] = 0.

        p_thresh = np.ma.masked_equal(p_thresh, 0.)

        return np.invert(p_thresh.mask), num_clust-sub  

    def get_fdr_threshold(self, p_vals, alpha, n_comparisons):

        p_sorted = (np.sort(p_vals.flatten()))
        k = [ii*alpha/n_comparisons for ii in range(1, n_comparisons+1)]

        below = p_sorted < k
        if np.sum(below) > 0:
            max_below = np.max(np.where(below)[0])
        else:
            max_below = 0

        print('%i tests passed thresholding'%(max_below))

        return k[max_below]

def compute_difference(map1, map2, n1, n2):
    ''' Given two covariance maps, compute the difference stat map.
    '''

    map1_z, map2_z = np.arctanh(map1), np.arctanh(map2)

    z_diff = (map1_z - map2_z) / np.sqrt((1/n1) + (1/n2))

    return z_diff

def dice(image1, image2, threshold=0.4):
    ''' Compute DICE score between two data matrices.
    '''
    image1[image1 > threshold] = 1.
    image1[image1 <= threshold] = 0.

    image2[image2 > threshold] = 1.
    image2[image2 <= threshold] = 0.    
    print(image2.astype(int))
    tru_p = np.sum(image1[image2.astype(int)])
    fal_p = np.sum(image1) - tru_p
    fal_n = np.sum(np.invert(image1[image2.astype(int)]))
    
    return 2*(tru_p) / (2*tru_p + fal_p + fal_n)

def normalized_mutual_information(image1, image2):
    """ Mutual information for joint histogram. Borrowed from
    Matthew Brett at 
    matthew-brett.github.io/teaching/mutual_information.html.
    """

    hgram, x_edges, y_edges = np.histogram2d(
        image1, 
        image2,
        bins=5000)

    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    nzs_x = px > 0
    nzs_y = py > 0
    norm = np.sqrt((-np.sum(px[nzs_x]*np.log2(px[nzs_x]))*(-np.sum(py[nzs_y]*np.log2(py[nzs_y])))))
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
