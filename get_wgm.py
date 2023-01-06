import numpy as np
import nibabel as nib
import sys
import os
from utils import largestAfterOpen
from scipy import ndimage


out_path = sys.argv[1]
prefix = sys.argv[2]
if out_path[-1]!='/':
	out_path = out_path+'/'
	

brain_mask = nib.load(out_path+prefix+'brain_mask.nii.gz').get_fdata().astype(bool)
affine = nib.load(out_path+prefix+'brain_mask.nii.gz').affine

img = nib.load(out_path+prefix+'t1_head_bc.nii.gz').get_fdata()
WM_prob = nib.load(out_path+prefix+'WM_prob.nii.gz').get_fdata()
GM_prob = nib.load(out_path+prefix+'GM_prob.nii.gz').get_fdata()
other_prob = nib.load(out_path+prefix+'other_prob.nii.gz').get_fdata()

other_mask = (ndimage.gaussian_filter(other_prob,sigma=1)>0.5) & brain_mask
wgm_mask = brain_mask & (~other_mask)
nib.save(nib.Nifti1Image(other_mask.astype(np.int16),affine),out_path+prefix+'other_mask.nii.gz')
nib.save(nib.Nifti1Image(wgm_mask.astype(np.int16),affine),out_path+prefix+'wgm_mask.nii.gz')
img[~wgm_mask]=0
nib.save(nib.Nifti1Image(img,affine),out_path+prefix+'wgm.nii.gz')
sys.exit()


