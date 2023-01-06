import numpy as np
import nibabel as nib
import sys
import os
from utils import largestAfterOpen
from scipy import ndimage

input_name = sys.argv[1]
stem_name = sys.argv[2]
atlas_path = sys.argv[3]
if atlas_path[-1] != '/':
	atlas_path = atlas_path+'/'	
valid_roi = nib.load(atlas_path+'valid_roi.nii').get_fdata().astype(bool)
valid_roi = ndimage.morphology.binary_dilation(valid_roi,structure=np.ones((3,3,3)),iterations=2)
affine = nib.load(atlas_path+'valid_roi.nii').affine

img = nib.load(input_name).get_fdata()
img_stem = np.array(img)
img_stem[valid_roi]=0
stem = largestAfterOpen(img_stem.astype(bool),iterations=2,structure=np.ones((3,3,3)),n_components=1)[0]
nib.save(nib.Nifti1Image(stem.astype(np.int16),affine),stem_name)
