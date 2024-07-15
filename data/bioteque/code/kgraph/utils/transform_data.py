import sys
import os
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, rankdata, zscore
import copy
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

#Quantile Normalization functions...
def get_quantile_ranks(matrix,features_as_columns=True):
    if features_as_columns:
        return np.nanmean(np.sort(np.asarray(matrix),axis=1), axis=0)
    else:
        return np.nanmean(np.sort(np.asarray(matrix),axis=0), axis=1)

def apply_quantile_norm(matrix,rank_mean, combine_rank_collisions_with='min', features_as_columns=True):
    m = np.asarray(matrix)
    m_transf = []

    if not features_as_columns:
        m = m.T

    for i in tqdm(range(np.shape(m)[0]), desc='Quantile Norm (if tie: %s)'%str(combine_rank_collisions_with)):

        if combine_rank_collisions_with =='min':
            v = rank_mean[rankdata(m[i],'min')-1]

        elif combine_rank_collisions_with =='average':
            r_lb = rankdata(m[i],'min') -1
            r_ub = rankdata(m[i],'max') -1
            v = np.asarray([rank_mean[r_lb[ix]:r_ub[ix]+1].mean() for ix in range(len(rank_mean))])

        elif combine_rank_collisions_with == 'max':
            v = rank_mean[rankdata(m[i],'max')-1]

        elif combine_rank_collisions_with is None:
            v = np.zeros(len(rank_mean))
            v[np.argsort(m[i])] = rank_mean
        else:
            sys.exit('Invalid "combine_repetition_with". Must be "average", "min" or "max" (not: %s)\n'%combine_rank_collisions_with)
        m_transf.append(v)

    m_transf = np.asarray(m_transf)

    if not features_as_columns:
        m_transf = m_transf.T

    if type(matrix) == pd.core.frame.DataFrame:
        m_transf = pd.DataFrame(m_transf, columns=matrix.columns, index=matrix.index.values)

    return m_transf

def quantile_norm(matrix, combine_rank_collisions_with='min', features_as_columns=True):
    return apply_quantile_norm(matrix,get_quantile_ranks(matrix,features_as_columns=features_as_columns),
                               features_as_columns=features_as_columns,
                               combine_rank_collisions_with=combine_rank_collisions_with)

def detect_outliers(v, method='zscore', zscore_cutoff= 3, iqr_margin = 1.5):

    if method == 'zscore':
         return zscore(v) > zscore_cutoff

    elif method == 'IQR':
        Q1 = np.quantile(v,0.25)
        Q3 = np.quantile(v,0.75)
        IQR = Q3 - Q1

        lbound = Q1 - iqr_margin*IQR
        ubound = Q3 + iqr_margin*IQR
        return np.array([x < lbound or x > ubound for x in v])

    else:
        sys.exit('Unknown method "%s"'%method)

def find_outliers_from_edges(edges, undirected, method='zscore', zscore_cutoff= 3, iqr_margin = 1.5):

    if undirected:
        v = np.array(edges).ravel()
        v  = Counter(v).items()
        return np.array([x[0] for x in v])[detect_outliers([x[1] for x in v],
                                                           method=method,
                                                           zscore_cutoff= zscore_cutoff,
                                                           iqr_margin = iqr_margin)]

    else:
        #--n1
        v = [x[0] for x in edges]
        v  = Counter(v).items()
        n1_outliers =  np.array([x[0] for x in v])[detect_outliers([x[1] for x in v],
                                                                   method=method,
                                                                   zscore_cutoff= zscore_cutoff,
                                                                   iqr_margin = iqr_margin)]

        #--n2
        v = [x[1] for x in edges]
        v  = Counter(v).items()
        n2_outliers =  np.array([x[0] for x in v])[detect_outliers([x[1] for x in v],
                                                           method=method,
                                                           zscore_cutoff= zscore_cutoff,
                                                           iqr_margin = iqr_margin)]

        return n1_outliers, n2_outliers

#----------------------

def merge_duplicated_cells(inputDF, axis, method):

    #Axis
    if axis in set(['column','Column',1]):
        inputDF = inputDF.groupby(inputDF.columns, axis=1)
    elif axis in set(['row','Row',0]):
        inputDF = inputDF.groupby(level=0, axis=0)

    #Method
    if method == 'mean':
        inputDF = inputDF.mean()

    elif method == 'median':
        inputDF = inputDF.median()

    elif method == 'max':
        inputDF = inputDF.max()

    elif method == 'min':
        inputDF = inputDF.min()

    else:
        sys.exit('Invalid method "%s"'%method)

    return(inputDF)

def standardize_prob(matrix, features_as_columns = True, negative_range=False):
    """Standarized function adapted from Harmonizome to standardize continuous data.

    If negative_range=True it scales the data from -1 to 1
        -IMPORTANT: Take into account that the tails 0.9/0.1 will be equivalent to the tails 0.95/0.05 of the NO-negative range distribution
    """
    m = np.asarray(copy.copy(matrix), dtype=np.float)

    if not features_as_columns:
        m = m.T

    N = m.shape[1]
    for i in tqdm(range(N), desc='standardizing features'):

        #Applying ECDF
        ourECDF = ECDF(m[:,i])
        m[:,i] = ourECDF(m[:,i])

        #Substracting mean
        m[:,i] = m[:,i] - m[:,i].mean()

    #Applying ECDF to global matrix
    values = m.ravel()
    ourECDF = ECDF(values)
    m = ourECDF(values).reshape(np.shape(m))

    #If negative_range it substracts the mean and doubles the probabilities
    if negative_range:
        m = 2 * (m - m.ravel().mean())

    #Final procedures
    if not features_as_columns:
        m = m.T
    if type(matrix) == pd.core.frame.DataFrame:
        m = pd.DataFrame(m, index=matrix.index.values, columns=matrix.columns)

    return m

def stratify_matrix(matrix, lower_bound=0.05, upper_bound=0.95):

    m = np.asarray(matrix, dtype=np.float)
    stratified_matrix = np.full(np.shape(m), 0)
    stratified_matrix[m<=lower_bound] = -1
    stratified_matrix[m>=upper_bound] = 1

    if type(matrix) == pd.core.frame.DataFrame:
        stratified_matrix = pd.DataFrame(stratified_matrix, columns = matrix.columns, index= matrix.index.values)
    return stratified_matrix

def drug_sens_stratified_waterfall(vector, uncertainty_margin=0.2, uncertainty_margin_if_lreg_fits=0.2, apply_uncertainty_at = None,
                                   min_cls = 0.01,  max_cls=1.0, min_sens_cutoff=None, global_cutoff=None, corr_cutoff = 0.95,
                                   ternary=False, return_cutoff=False, plot=False):

    """
    Given a vector of drug_sensitivity across CLs it returns the stratified (0/1 or -1/0/1) using the waterfall method.
    NaN values are conserved.
    It expects IC50 or AUC matrix so lower numbers must indicate higher sensitivity.

    Keywords:

    uncertainty_margin -- proportion of the cutoff used as margin of uncertainty (neither sensitive nor resistant)
    uncertainty_margin_if_lreg_fits -- proportion of the cutoff used as margin if the waterfall distribution shape is a straight line
    min_cls -- TOP number (int) or proportion (float) of cls to be considered as sensitive for each drug (default= TOP 1% of sensitive CLS)
    apply_uncertainty_at -- sensitivity value border to to apply the uncertainty cutoff (i.e. when using AUC, 0.8 is a good cutoff)
    min_sens_cutoff -- Minimum sensitivity to consider a CL sensitive (i.e. when using AUC, 0.9 is a good cutoff)
    global_cutoff -- If given and when the pearson correlation fails(==NaN) (usually as all the cls have the same sensitivity) it will use it as a cutoff.
    corr_cutoff -- Pearson correlation cutoff used to decide whether the waterfall is a straight line or not.
    ternary-- values within the uncertainty margin range are returned as "0". Values lower than lower bound are "1" while values higher than upper bound are "-1"
    return_cutoff -- If true it returns a tuple with the vector and the cutoff used
    plot -- If True it plots the waterfall curve with the cutoffs used for the drug
    """

    #0) Checking parameters
    v = np.asarray(vector)
    if ternary and uncertainty_margin == 1:
        sys.exit('For ternary stratification yoy need to provide an uncertainty margin greater than 1')

    #1) Sort values
    y = np.sort(v[~pd.isnull(v)])
    x = np.arange(1,len(y)+1,1).reshape(-1,1)


    #2) Fit linear regression
    lrg = LinearRegression()
    lrg.fit(x.reshape(-1,1),y)

    #3) Predict
    y_pred = lrg.predict(x)

    #4) Get correlation
    R = pearsonr(y,y_pred)[0]

    #5) Get binarization cutoff

    #--5.1) If < corr_cutoff get more distant point to the line
    if R < corr_cutoff:
        def get_y_straight(x,y):
            m = (y[0]-y[-1])/(x[0]-x[-1]) #Slope
            b = (x[0]*y[-1] - x[-1]*y[0])/(x[0]-x[-1]) # Y-Intercept
            y_straight = m*np.asarray(x) + b
            #--Force initial and final point to be the same
            y_straight[0] = y[0]
            y_straight[-1] = y[-1]
            return y_straight.ravel()

        y_straight = get_y_straight(x,y)
        cutoff = y[np.argmax(abs(y-y_straight))]

        #Prevents a bug when there are ties
        if cutoff == y[-1]:
            cutoff = 0
            
    #--5.2) if >= corr_cutoff get median of auc as cutoff point
    elif R >= corr_cutoff:
        cutoff = np.median(y)
        uncertainty_margin = uncertainty_margin_if_lreg_fits
           
    elif pd.isnull(R): #Usually because all cls has the same value. This could be solved with a global cutoff...
        if global_cutoff:
            cutoff = global_cutoff
        else:
            cutoff = y[0] - 1 #No cll will be considered as sensitive

    #6) Stratifying
    np.warnings.filterwarnings('ignore')
    str_v = v.copy()

    if ternary:
        if apply_uncertainty_at is not None and R < corr_cutoff and cutoff < apply_uncertainty_at:
            cutoff_lb = cutoff
        else:
            cutoff_lb = cutoff - abs(cutoff*uncertainty_margin)
        if apply_uncertainty_at is not None and R < corr_cutoff and cutoff > apply_uncertainty_at:
            cutoff_ub = cutoff
        else:
            cutoff_ub = cutoff + abs(cutoff*uncertainty_margin)
            
        str_v[v<cutoff_lb] = 1
        str_v[v>cutoff_ub] = -1
        str_v[(v>=cutoff_lb) & (v<=cutoff_ub)] = 0
    else:
        if apply_uncertainty_at is not None and R < corr_cutoff and cutoff < apply_uncertainty_at:
            pass
        else:
            cutoff = cutoff - abs(cutoff*uncertainty_margin)
        str_v[v<cutoff] = 1
        str_v[v>=cutoff] = 0
            
    if min_sens_cutoff:
        if ternary:
            str_v[v >= min_sens_cutoff] = -1
        else:
            str_v[v >= min_sens_cutoff] = 0
            
    #Applying min_cls
    if min_cls:
        if type(min_cls) == float:
            min_cls = int(np.round(sum(~pd.isnull(v))*min_cls))
        
        if (str_v==1).sum() < min_cls:
            str_v[[ix for ix in np.argsort(v)[:min_cls] if not pd.isnull(v[ix])]] = 1
    
    #Applying max_cls
    if max_cls:
        if type(max_cls) == float:
            max_cls = int(np.round(sum(~pd.isnull(v))*max_cls))
        
        if (str_v==1).sum() > max_cls:
            str_v[[ix for ix in np.argsort(v)[max_cls:] if str_v[ix] == 1]] = 0

    np.warnings.filterwarnings('default')

    if plot:
        plt.figure(figsize=(5,5),dpi=100)
        plt.plot(x,y)
        if R < corr_cutoff:
            plt.plot(x,y_straight)
        if ternary:
            plt.axhline(cutoff_lb, color='black',linestyle='--', label='lb %.2f'%cutoff_lb)
            plt.axhline(cutoff_ub, color='black',linestyle='-.', label='ub %.2f'%cutoff_ub)
        else:
            plt.axhline(cutoff, color='black',linestyle=':', label='%.2f'%cutoff)

        if min_cls:
            plt.axvline(min_cls, linestyle=':', color='black', label='min_cl')
        if max_cls < int(np.round(sum(~pd.isnull(v)))):
         
            plt.axvline(max_cls, linestyle=':', color='purple', label='max_cl')

        plt.legend()
        plt.ylabel('Sensitivity')
        plt.xlabel('# Cell lines')
        plt.show()
        plt.close()

    if return_cutoff:
        return str_v, cutoff
    else:
        return str_v

