import numpy as np
import nibabel as nib
import sys
from scipy.ndimage import gaussian_filter

n_masks = len(sys.argv)-3
output_name = sys.argv[-2]
smooth = (sys.argv[-1] == 'y')
affine = nib.load(sys.argv[1]).affine
combined = np.zeros(nib.load(sys.argv[1]).get_fdata().shape)
for i in range(1,n_masks+1):
	m = nib.load(sys.argv[i]).get_fdata()>0.5
	combined[m]=i
if smooth:
	combined = gaussian_filter(combined.astype(float), sigma=1)
nib.save(nib.Nifti1Image(combined,affine),output_name)
