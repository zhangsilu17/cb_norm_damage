import numpy as np
import nibabel as nib
from utils import largestAfterOpen
from scipy import ndimage
import sys



out_path = sys.argv[1]
suit_path = sys.argv[2]
prefix = sys.argv[3]
comp_ls = sys.argv[4]
if out_path[-1]!='/':
	out_path = out_path+'/'
if suit_path[-1]!='/':
	suit_path = suit_path+'/'
comp_ls = comp_ls.split(',')
if comp_ls[0]=='':
	comp_ls = []
else:
	comp_ls = [int(i) for i in comp_ls]

img_norm = nib.load(out_path+prefix+'t1_norm.nii.gz').get_fdata()
affine = nib.load(out_path+prefix+'t1_norm.nii.gz').affine
valid_roi = nib.load(suit_path+'valid_roi.nii').get_fdata().astype(bool)
structure=ndimage.generate_binary_structure(3,1)
valid_roi = ndimage.morphology.binary_erosion(valid_roi,structure=structure,iterations=1)
diff = valid_roi & (img_norm < 50)

damage_ls = largestAfterOpen(diff,iterations=0,n_components=3)
if damage_ls is None:
	print('No damge detected')
	sys.exit(0)
n_roi = len(damage_ls)
for i in range(n_roi):
	# print(ndimage.measurements.center_of_mass(damage_ls[i]))
	nib.save(nib.Nifti1Image(damage_ls[i].astype(np.int16),affine),out_path+prefix+'damage_'+str(i)+'.nii.gz')

print(f'Use {comp_ls} as damage')
damage = np.zeros(img_norm.shape).astype(bool)
for i in comp_ls:
	damage = damage | damage_ls[i]

damage_size = damage.sum()
print('damage size: '+str(damage_size))
if damage_size < 5:
	print('Size too small, no damage ROI detected')
	nib.save(nib.Nifti1Image(np.zeros(img_norm.shape).astype(np.int16),affine),out_path+prefix+'damage.nii.gz')
else:
	nib.save(nib.Nifti1Image(damage.astype(np.int16),affine),out_path+prefix+'damage.nii.gz')

