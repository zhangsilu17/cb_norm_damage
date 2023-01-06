import numpy as np
import nibabel as nib
from utils import largestAfterOpen
from scipy import ndimage
import sys


out_path = sys.argv[1]
prefix = sys.argv[2]
th = int(sys.argv[3])
if out_path[-1]!='/':
	out_path = out_path+'/'
structure=ndimage.generate_binary_structure(3, 1)
img = nib.load(out_path+prefix+'t1_head_bc.nii.gz').get_fdata()
cb_temp_msk = nib.load(out_path+prefix+'temp_cb_msk.nii.gz').get_fdata().astype(bool)
cb_temp_msk = ndimage.morphology.binary_closing(cb_temp_msk,structure,iterations=2)
cb_msk = np.array(cb_temp_msk)
affine = nib.load(out_path+prefix+'temp_cb_msk.nii.gz').affine
wgm_msk = nib.load(out_path+prefix+'wgm_mask.nii.gz').get_fdata().astype(bool)
csf_prob = nib.load(out_path+prefix+'other_prob.nii.gz').get_fdata()
gm_prob = nib.load(out_path+prefix+'GM_prob.nii.gz').get_fdata()
if th < 0:
	cavity = np.zeros(img.shape).astype(bool)
else:
	print('cavity th: '+str(th))
	cavity_th = cb_msk & (img<th)
	cavity = ndimage.morphology.binary_opening(cavity_th,structure,iterations=1)
	vol_cavity = cavity.sum()
	print('cavity size: '+str(vol_cavity))

th_l = np.percentile(img[wgm_msk],4)
th_h = np.percentile(img[wgm_msk],98)*1.2


cb_msk[~wgm_msk]=False

cb_msk[cavity]=False

cb_msk = largestAfterOpen(cb_msk,iterations=2,structure=structure,n_components=1)[0]


# Add missed peduncle due to opening
z_top = np.where(np.sum(np.sum(cb_msk,axis=1),axis=0)>0)[0].max()-5
z_down = z_top-30
y_front = np.where(np.sum(np.sum(cb_msk,axis=2),axis=0)>0)[0].max()-20
y_back = y_front-20
brain_mask = nib.load(out_path+prefix+'brain_mask.nii.gz').get_fdata().astype(bool)
x_mid = int((np.where(np.sum(np.sum(brain_mask,axis=2),axis=1)>0)[0].min()+np.where(np.sum(np.sum(brain_mask,axis=2),axis=1)>0)[0].max())/2)
missing_z = np.zeros(img.shape).astype(bool)
missing_y = np.zeros(img.shape).astype(bool)
missing_x = np.zeros(img.shape).astype(bool)
missing_z[:,:,z_down:z_top]=True
missing_y[:,y_back:y_front,:]=True
missing_x[(x_mid-9):(x_mid+10),:,:]=True
missing_roi = (missing_x & missing_y) & missing_z

nib.save(nib.Nifti1Image(missing_roi.astype(np.int16),affine),out_path+prefix+'peduncle_bounding.nii.gz')

peduncle_roi = missing_roi & (img>th_l) & (img<th_h)
cb_msk[peduncle_roi]=True
cb_msk = largestAfterOpen(cb_msk,iterations=0,structure=structure,n_components=1)[0]
cb_msk_close = ndimage.morphology.binary_closing(cb_msk,structure,iterations=5)
cb_msk_diff = cb_msk_close & (~cb_msk)
cb_msk_miss = cb_msk_diff & (img>th_l) & (img<th_h)
cb_msk = cb_msk | cb_msk_miss
cb_msk = ndimage.gaussian_filter(cb_msk.astype(float),sigma=1)>0.5
cb_msk[peduncle_roi]=True
cb_msk = largestAfterOpen(cb_msk,iterations=0,structure=structure,n_components=1)[0]

nib.save(nib.Nifti1Image(cb_msk.astype(np.int16),affine),out_path+prefix+'cb_mask.nii.gz')

wm_prob = nib.load(out_path+prefix+'WM_prob.nii.gz').get_fdata()
wm_prob[~cb_msk] = 0
nib.save(nib.Nifti1Image(wm_prob,affine),out_path+prefix+'cb_wm.nii.gz')
gm_prob[~cb_msk] = 0
nib.save(nib.Nifti1Image(gm_prob,affine),out_path+prefix+'cb_gm.nii.gz')
