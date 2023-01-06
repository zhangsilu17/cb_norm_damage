import sys
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import median_filter
from utils import getCSFprior,normalizeProb,tumorDetection,saveMaps,largestAfterOpen
from scipy import ndimage
import random
from shutil import copy2


n_tissue=int(sys.argv[1]) #number of tissue types to model. 3: WM, GM, other; 4: WM, GM, CSF, other
print(f'n_tissue: {n_tissue}')
if n_tissue not in [3,4]:
	print('Number of tissue type must be 3 or 4')
	print(f'Value given: {n_tissue}')
	exit(1)
in_path=sys.argv[2]
prefix=sys.argv[3]
if in_path[-1] != '/':
	in_path = in_path+'/'
print("input path: "+in_path)
out_path = in_path+'probability/'
if not os.path.exists(out_path):
   os.makedirs(out_path)

# Load brain mask
brain_mask_name = glob.glob(in_path+'*brain_mask.nii.gz')[0]
brain_mask = nib.load(brain_mask_name).get_fdata().astype(bool)


suffix = '_head_bc.nii.gz'
contrast_path_ls = glob.glob(in_path+'*'+suffix)
contrasts = [p.split('/')[-1].replace(suffix,'').replace(prefix,'') for p in contrast_path_ls]

print('contrasts available')
print(contrasts)


affine = nib.load(contrast_path_ls[0]).affine

if not os.path.exists(out_path+'iter 0/'):
		os.makedirs(out_path+'iter 0/')
print('Loading input images')
input_images = {}

for C in contrasts:
	original_img = nib.load(in_path+prefix+C+suffix).get_fdata()
	input_images[C]=original_img


prior_map = {}
WM_prior = nib.load(in_path+prefix+'WM_prior.nii.gz').get_fdata()
WM_prior[~brain_mask]=0
WM_prior = WM_prior/WM_prior.max()
GM_prior = nib.load(in_path+prefix+'GM_prior.nii.gz').get_fdata()
GM_prior[~brain_mask]=0
GM_prior = GM_prior/GM_prior.max()
csf_img_ls = ['flair','t1','t1post','adc']
if n_tissue==4:
	print('Estimate CSF prior')
	for csf_img in csf_img_ls:
		if csf_img in contrasts:
			found_csf_img = True
			break

	if not found_csf_img:
		print(f'At least one of {csf_img_ls} must be available to estimate CSF prior')
		exit(1)

	print(f'Using {csf_img} to estimate CSF prior')
	csf_prior = getCSFprior(input_images[csf_img],brain_mask,affine,mr_type=csf_img,out_path=out_path)
	prior_map['CSF'] = csf_prior
	WM_prior[csf_prior>0.1]=0.01
	GM_prior[csf_prior>0.1]=0.01

prior_map['WM'] = WM_prior
prior_map['GM'] = GM_prior


# Normalize priors
prior_map = normalizeProb(prior_map,brain_mask)
if n_tissue==4:
	other_prior=1/10*(prior_map['WM']+prior_map['GM'])
else:
	other_prior = np.zeros(brain_mask.shape)
	other_prior[brain_mask]=1
prior_map['other'] = other_prior
prior_map = normalizeProb(prior_map,brain_mask)

saveMaps(prior_map,out_path+'iter 0/',affine)

# Run outlier segmentation
n_iters = 3
print('Contrasts used for whole tumor segmentation:')
print(input_images.keys())
if len(input_images.keys())==0:
	sys.exit()
print('Brain tissues under modeling:')
print(prior_map.keys())

bw_max = np.inf
bw_min = 2
prior_map = tumorDetection(prior_map,brain_mask,input_images,n_iters,out_path,affine,1,bw_max=bw_max,bw_min=bw_min)
probability_files = glob.glob(out_path+'posterior/iter '+str(n_iters)+'/*prob.nii.gz')
for f in probability_files:
	copy2(f, in_path+prefix+f.split('/')[-1])
os.sys.exit(0)

