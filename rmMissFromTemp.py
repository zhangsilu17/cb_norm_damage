import numpy as np
import nibabel as nib
import sys
import os
from utils import largestAfterOpen
from scipy import ndimage

img_name = sys.argv[1]
template_name = sys.argv[2]
roi_th = int(sys.argv[3])
print(f'roi_th: {roi_th}')
f_name = img_name.split('/')[-1]
out_path = img_name.replace(f_name,'')
img = nib.load(img_name).get_fdata()
affine = nib.load(template_name).affine
template = nib.load(template_name).get_fdata()
bi_th = 50
template_bi = template>bi_th
template_bi[img>50]=0
missing_roi = largestAfterOpen(template_bi,iterations=3,structure=np.ones((3,3,3)),n_components=1)
if missing_roi is not None:
	missing_roi=missing_roi[0]
else:
	missing_roi=np.zeros(img.shape).astype(bool)
print(f'missing roi size: {missing_roi.sum()}')
# roi_th = 15000
if missing_roi.sum()<roi_th:
	print(f'Missing roi < {roi_th}, do not change template')
	temp_valid = template>bi_th
else:
	temp_valid = (~missing_roi) & (template>bi_th)
nib.save(nib.Nifti1Image(missing_roi.astype(np.int16),affine),out_path+'temp_missing_roi.nii.gz')
nib.save(nib.Nifti1Image(temp_valid.astype(np.int16),affine),out_path+'temp_valid_roi.nii.gz')
