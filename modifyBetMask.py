import nibabel as nib
import sys
from scipy import ndimage
import numpy as np

input_name = sys.argv[1]
output_name = sys.argv[2]
close_iter = int(sys.argv[3])

mask = nib.load(input_name).get_fdata().astype(bool)
affine = nib.load(input_name).affine
mask_mc = ndimage.morphology.binary_closing(mask,structure=np.ones((3,3,3)).astype(bool),iterations=close_iter)
mask_fh=ndimage.morphology.binary_fill_holes(mask_mc)
nib.save(nib.Nifti1Image(mask_fh.astype(np.int16),affine),output_name)