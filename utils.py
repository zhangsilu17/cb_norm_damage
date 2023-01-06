import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.covariance import MinCovDet
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelmax,argrelmin
import os
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import random
from scipy.ndimage import gaussian_filter
from scipy import ndimage



def plotSlice(img,n_slices=6,positions=None,mask_ls=None,direction='z',file_name=None,title=None,vmin=0,vmax=None,alpha=0.5,
	tight_layout=True,color_ls=['red_transparent_full_alpha_range','green_transparent_full_alpha_range'],fig_size=10,background=0,crop=False):
	if mask_ls is None:
		mask = img.astype(bool)
		mask_ls = []
	else:
		mask = mask_ls[0]
	if direction == 'x':
		axis_sum = np.sum(np.sum(mask,axis=2),axis=1)
	elif direction == 'y':
		axis_sum = np.sum(np.sum(mask,axis=2),axis=0)
	else:
		axis_sum = np.sum(np.sum(mask,axis=1),axis=0)
# 	print(axis_sum)
	idx_ls = np.where(axis_sum>0)[0]
# 	print(idx_ls)
	step = max(int((idx_ls.max()-idx_ls.min())/(n_slices+1)),1)
# 	print('step: '+str(step))
	start_idx = idx_ls.min()+step
# 	print('start: '+str(start_idx))
	fig, ax = plt.subplots(1,n_slices,figsize=[fig_size*n_slices,fig_size])
	for i in range(n_slices):
		if positions is None:
			idx = start_idx+i*step
		else:
			idx = positions[i]
		# print(f'idx: {idx}')
		if n_slices == 1:
			axis = ax
		else:
			axis = ax[i]
		if direction == 'x':
			s = np.rot90(img[idx,:,:])
			mask_s_ls = [np.rot90(m[idx,:,:]) for m in mask_ls] 
		elif direction == 'y':
			s = np.flip(np.rot90(img[:,idx,:]),axis=1)
			mask_s_ls = [np.flip(np.rot90(m[:,idx,:]),axis=1) for m in mask_ls] 
		else:
			s = np.rot90(img[:,:,idx])
			mask_s_ls = [np.rot90(m[:,:,idx]) for m in mask_ls] 
		if crop:
			s,b = crop2DImg(s,background=background)
			if mask_ls is not None:
				mask_s_ls = [crop2DImg(m,bouding_box=b)[0] for m in mask_s_ls]
		axis.imshow(s,'gray',filternorm=False,vmin=vmin,vmax=vmax)

		for m in range(len(mask_s_ls)):
			axis.imshow(mask_s_ls[m],color_ls[m],alpha=alpha,filternorm=False)

	for i in range(n_slices):
		axis.axis('off')
	fig.subplots_adjust(wspace=0,hspace=0)
	if title is not None:
		plt.title(title,fontsize=15)
	if tight_layout:
		plt.savefig(file_name,bbox_inches='tight',facecolor='k')
	else:
		plt.savefig(file_name,facecolor='k')
	plt.close()

def largestAfterOpen(binary_img,iterations,structure=None,n_components=None,size_min=1):
	if iterations>0:
		binary_img = ndimage.morphology.binary_erosion(binary_img,structure,iterations=iterations)
	labeled_array, num_features = ndimage.measurements.label(binary_img)
	CCsize = np.zeros((num_features+1,)).astype(int)
	for c in range(1,num_features+1):
		CCsize[c] = np.sum(labeled_array==c)
	idx_sort = np.argsort(-CCsize)
	# print(CCsize[idx_sort])
	# print(idx_sort)
	if n_components == None:
		n_components = sum(CCsize>=size_min)

	n_components = min(n_components,num_features)
	if n_components==0:
		print(f'No connected components with size >= {size_min}')
		return None
	largest = []
	for n in range(n_components):
		idx = idx_sort[n]
		size = CCsize[idx]
		# print('componet size: '+str(size))
		roi =labeled_array==idx
		if iterations>0:
			roi = ndimage.morphology.binary_dilation(roi,structure,iterations=iterations)
		largest.append(roi)
	return largest


def pad_closing(binary_img,iterations,structure):
	if (iterations % 2)==1:
		n = iterations+1
	else:
		n = iterations
	img_size = np.array(binary_img.shape)
	structure_size = np.array(structure.shape)
	print(img_size)
	print(structure_size)
	print(n)
	pad_img = np.zeros(n*structure_size+img_size)
	print('padded image shape: '+str(pad_img.shape))
	m = int(n/2)
	pad_img[m*structure_size[0]:(img_size[0]+m*structure_size[0]),m*structure_size[1]:(img_size[1]+m*structure_size[1]),m*structure_size[2]:(img_size[2]+m*structure_size[2])]=binary_img
	pad_img = ndimage.binary_closing(pad_img, structure=structure, iterations=iterations)
	img = pad_img[m*structure_size[0]:(img_size[0]+m*structure_size[0]),m*structure_size[1]:(img_size[1]+m*structure_size[1]),m*structure_size[2]:(img_size[2]+m*structure_size[2])]
	return img
def peakNorm(img,brain_mask,file_name=None,mr_type='t1'):
	# Identify GM peak or (WM/GM) peak if there are not separate peaks for WM and GM. Use this peak as reference to find a th to separate CSF from GM/WM 
	contrast_th = {'t1':0.7,'t1post':0.7,'flair':0.5,'dwi':0.7,'t2':1.7,'b0':1.5,'adc':2}
	print('Normalizing image...')
	img = img.astype(float)
	bw = 0.04*np.percentile(img[brain_mask],90)
	print('Random sampling 2000 voxels')
	random.seed(10)
	kde_sample=np.array(random.sample(img[brain_mask].tolist(),2000))
	# print(len(kde_sample))
	kde = KernelDensity(bandwidth=bw)
	kde.fit(kde_sample.reshape(-1,1))
	pts = np.arange(round(np.percentile(img[brain_mask],99)))
	pts = pts.reshape(-1,1)
	log_density = kde.score_samples(pts).reshape((len(pts),))
	density = np.exp(log_density)
	density_modified = np.array(density)
	# Discard peak lower than l
	if mr_type in ['t1','t1post','flair'] or ('dwi' in str(mr_type)):
		l = int(np.percentile(img[brain_mask],25))
	else:
		l=0
	density_modified[:l]=0
	density_modified = density_modified/density_modified.max()
	maxima_intensity = argrelmax(density_modified)[0]
	minima_intensity = argrelmin(density)[0]
	maxima_density = density_modified[maxima_intensity]
	th = None
	has_maxima = False
	if (mr_type in ['t1','t1post','flair']) or ('dwi' in mr_type):
		peak_intensity = maxima_intensity[maxima_density>0.7].min()
		maxima_intensity = argrelmax(density)[0]
		if (mr_type != 'flair') & (len(minima_intensity)>0):
			minima = minima_intensity[minima_intensity<peak_intensity]
			maxima = maxima_intensity[maxima_intensity<peak_intensity]
			if (len(minima) > 0)  and (len(maxima) > 0):
				density_maxima = density[maxima].max()
				maxima = maxima[np.argmax(density[maxima])]
				minima = minima.max()
				if (density_maxima>0.1*density[peak_intensity]) and (maxima<minima):
					has_maxima = True
					th = minima
	elif mr_type in ['t2','adc','b0']:
		peak_intensity = maxima_intensity[maxima_density>0.8].max()
		maxima_intensity = argrelmax(density)[0]
		if len(minima_intensity)>0:
			minima = minima_intensity[minima_intensity>peak_intensity]
			maxima = maxima_intensity[maxima_intensity>peak_intensity]
			if (len(minima) > 0)  and (len(maxima) > 0):
				density_maxima = density[maxima].max()
				maxima = maxima[np.argmax(density[maxima])]
				minima = minima.min()
				if (density_maxima>0.1*density[peak_intensity]) and (maxima>minima):
					has_maxima = True
					th = minima

	else:
		raise ValueError(f'{mr_type} not recognized')

	print("peak intensity: "+str(peak_intensity))
	img_norm = img/peak_intensity
	if th is None:
		if 'dwi' in mr_type:
			th = contrast_th['dwi']*peak_intensity
		else:
			th = contrast_th[mr_type]*peak_intensity

	th = int(th)
	if file_name is not None:
		plt.plot(pts,density,c='k')
		plt.axvline(x=peak_intensity,c='k')
		plt.axhline(y=0.1*density[peak_intensity],c='k',ls='dotted')
		plt.axvline(x=th,c='r',ls='--')
		if l>0:
			plt.axvline(x=l,c='b',ls='--')
			plt.text(l+2, density.max()/2, s=f'l={l}',fontsize=8)
		plt.text(th+2, density.max()/3, s=f'th={th}',fontsize=8)
		plt.text(peak_intensity+2, density.max()/4, s=f'peak={peak_intensity}',fontsize=8)
		if has_maxima:
			plt.axvline(x=maxima,c='g')
		plt.savefig(file_name)
		plt.close()

	return img_norm,peak_intensity,th

def setPrior(img,brain_mask,affine,mr_type='flair',out_path=None):
	map_unit = np.zeros(brain_mask.shape)
	map_unit[brain_mask] = 1
	prior_map = {}
	prior_map['WM'] = 0.5 * map_unit
	prior_map['GM'] = 0.5 * map_unit
	csf_prior = getCSFprior(img,brain_mask,affine,mr_type,out_path)
	prior_map['CSF'] = csf_prior
	prior_map= normalizeProb(prior_map,brain_mask)
	if out_path is not None:
		saveMaps(prior_map,out_path+'iter 0/',affine)
	return prior_map

def gauss(x,mu,sigma):
	p = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/2/sigma**2)
	return p
def bimodal(x,mu1,sigma1,p1,mu2,sigma2,p2):
	p = p1*gauss(x,mu1,sigma1)+p2*gauss(x,mu2,sigma2)
	return p 
def multimodal(x,*params):
	n_model = int(len(params)/3)
	p=0
	for i in range(n_model):
		par_i = params[i*3:(i*3+3)]
		mu_i = par_i[0]
		sigma_i = par_i[1]
		p_i = par_i[2] 
		p = p+p_i*gauss(x,mu_i,sigma_i)
	return p
def normalizeProb(prob_map,brain_mask):
	prob_norm_map = {}
	all_classes = prob_map.keys()
	prob_sum = np.zeros(brain_mask.shape)
	for C in all_classes:
		prob_sum = prob_sum+prob_map[C]
	zero_sum = (prob_sum == 0)
	brain_mask = brain_mask & (~zero_sum)
	for C in all_classes:
		prob = prob_map[C]
		prob[brain_mask] = prob[brain_mask]/prob_sum[brain_mask]
		prob[np.isnan(prob)]=0
		prob[~brain_mask]=0
		prob_norm_map[C] = prob
	return prob_norm_map

def saveMaps(prob_map,out_path,affine):
	all_classes = prob_map.keys()
	for C in all_classes:
		prob=prob_map[C]
		nib.save(nib.Nifti1Image(prob,affine),out_path+C+'_prob.nii.gz')
		plotSlice(prob,file_name=out_path+C+' prob.png',title=C+' prob',vmin=0,vmax=1)

def samplingTrainData(prob,tau):
	prob_flat = prob.flatten()
	prob_flat[prob_flat<=tau]=0
	print('total popultation: '+str(sum(prob_flat>0)))
	n_sample = 2000
	np.random.seed(10)
	idx = np.random.choice(len(prob_flat),size=n_sample,replace=True,p=prob_flat/sum(prob_flat))
	return idx

def _getTissueSamples(p_map,C,brain_mask,input_images):
	p_map_copy = np.array(p_map)
	p_map_copy[~brain_mask]=0
	tau = {}
	tau['WM']=0.1
	tau['GM']=0.1
	tau['CSF']=0.5
	tau['other']=0
	idx = samplingTrainData(p_map,tau[C])
	print(C)
	print(f'# of samples: {len(idx)}')
	# Outlier detection for homo_classes
	homo_classes = ['WM','GM']
	contrasts = list(input_images.keys())
	if C in homo_classes:
		data = np.zeros((len(idx),len(contrasts)))
		for i in range(len(contrasts)):
			img = input_images[contrasts[i]]
			channel_sample = img.flatten()[idx]
			data[:,i] = channel_sample
		mcd = MinCovDet(support_fraction=0.8,random_state=0)
		try:
			mcd.fit(data)
			inliers = mcd.support_
			idx = idx[inliers]
		except ValueError:
			print('The covariance matrix of the support data is equal to 0, did not perform MCD to remove outliers')
	
	print(f'# of inliers: {len(idx)}')
	return idx

def getTissueSamples(prior_map,brain_mask,input_images,n_jobs=1):
	samples = {}
	all_classes = prior_map.keys()
	n_classes = len(all_classes)
	for C in all_classes:
		samples[C] = _getTissueSamples(prior_map[C],C,brain_mask,input_images)
	return samples

def fitMultiModal(modal_intensity,pts,density,sigma,mu_bounds):
	n_modal = len(modal_intensity)
	expected = []
	low_bound = []
	up_bound = []
	for m in range(n_modal):
		expected.append(modal_intensity[m])
		expected.append(sigma)
		expected.append(1/n_modal)
		low_bound = low_bound+[mu_bounds[0],0,0]
		up_bound = up_bound+[mu_bounds[1],pts.max()/4,1]

	params,cov=curve_fit(multimodal,pts,density,expected,bounds=(low_bound, up_bound))

	Y = np.zeros((len(pts),n_modal))
	Y_modal = np.zeros((len(pts),n_modal))
	modal_intensity = np.zeros((n_modal,))
	p = np.zeros((n_modal,))
	sigma_ls = np.zeros((n_modal,))
	for m in range(n_modal):
		par_m = params[m*3:(m*3+3)]
		mu_m = par_m[0]
		modal_intensity[m]=mu_m
		sigma_m = par_m[1]
		sigma_ls[m] = sigma_m
		p_m = par_m[2] 
		Y[:,m] = gauss(pts,mu_m,sigma_m)
		p[m] = p_m
		Y_modal[:,m] = p_m*gauss(pts,mu_m,sigma_m)
	return Y,modal_intensity,sigma_ls,p

def inRange(x,l,h):
	if ((x>l) and (x<h)):
		return True
	else:
		return False

def fitKDE(pts,n_classes,img,brain_mask,tissue_type,mr_type,img_sample=None,bw_max=np.inf,bw_min=1,norm_path=None):
	if norm_path is not None:
		norm_file = norm_path+mr_type+'_norm.png'
	else:
		norm_file = None
	img_norm,peak,th = peakNorm(img,brain_mask,file_name=norm_file,mr_type=mr_type) 

	if img_sample is None:
		img_sample = img[brain_mask]
	img_sample = img_sample[img_sample>0]
	img_max = np.percentile(img[brain_mask],95)
	# th_h=round(img_max*0.8)
	print(f'Fitting KDE for {mr_type}. th: {th}')
	if tissue_type == 'other':
		img_sample_kde = img_sample
		bw = img_sample_kde.max()*0.04
	else:
		if mr_type in ['flair','t1','t1post'] or 'dwi' in mr_type:
			bw = 0.03*(img_max-th)
			if tissue_type == 'CSF':
				bw=bw*2
				print(f'use intensity < {th} to estimate KDE for {mr_type}')
				img_sample_kde = img_sample[img_sample<th]
			elif tissue_type in ['WM','GM']:
				print(f'use intensity > {th} to estimate KDE for {mr_type}')
				img_sample_kde = img_sample[img_sample>th]
			else: 
				raise ValueError(f'{tissue_type} not defined')
		elif mr_type in ['t2','adc','b0']:
			bw = 0.01*th
			if tissue_type == 'CSF':
				bw = bw*2
				print(f'use intensity > {th} to estimate KDE for {mr_type}')
				img_sample_kde = img_sample[img_sample>th]
			elif tissue_type in ['WM','GM']:
				print(f'use intensity < {th} to estimate KDE for {mr_type}')
				img_sample_kde = img_sample[img_sample<th]
			else: 
				raise ValueError(f'{tissue_type} not defined')
		else:
			raise ValueError(f'{mr_type} not defined')
	if len(img_sample_kde)>2000:
		random.seed(10)
		img_sample_kde=np.array(random.sample(img_sample_kde.tolist(),2000))

	bw = min(bw,bw_max)
	bw = max(bw_min,bw)
	bw = round(bw)
	print(f'bw: {bw}')
	kde = KernelDensity(bandwidth=bw)
	kde.fit(img_sample_kde.reshape(-1,1))
	pts = pts.reshape(-1,1)
	log_density = kde.score_samples(pts).reshape((len(pts),))
	density = np.exp(log_density)
	mu_bounds = (img_sample_kde.min(),img_sample_kde.max())
	return density,th,bw,mu_bounds

def getCSFprior(img,brain_mask,affine,mr_type='flair',out_path=None):
	print("Estimating CSF")
	img_data_brain = img[brain_mask]
	pts = np.arange(round(img_data_brain.max()*1.1))
	density,th,bw,mu_bounds = fitKDE(pts,4,img,brain_mask,'CSF',mr_type)
	plt.plot(pts,density,c='k')
	plt.axvline(x=th,c='k')
	plt.title(f'{mr_type} bw={bw}')
	if out_path is not None:
		if not os.path.exists(out_path+'iter 0'):
			os.makedirs(out_path+'iter 0')
		plt.savefig(out_path+'iter 0/csf_density.png')
	plt.close()
	prob = density/density.max()
	csf_prior = np.zeros(brain_mask.shape)
	csf_prior[brain_mask]=prob[np.round(img_data_brain).astype(int)]
	nib.save(nib.Nifti1Image(csf_prior,affine),out_path+'iter 0/csf_prior.nii.gz')
	return csf_prior

def updatePosterior(prior_map,brain_mask,sample_map,img,affine,out_path,it,bw_max=np.inf,bw_min=1,WM_GM_contrast=None,smooth=False):
	if WM_GM_contrast is None:
		WM_GM_contrast={'t1':[1.5,2.2],'t1post':[1.3,2],'flair':[0.6,0.85],'adc':[0.85,0.95],'t2':[0.6,0.8],'dwi':[0.6,0.8],'b0':[0.6,0.8]}

	p_min={'WM':{1:0.2,2:0.25,3:0.3},'GM':{1:0.25,2:0.3,3:0.4}}
	con = out_path.split('/')[-2]
	print('')
	print('Update posterior for '+con)
	if not os.path.exists(out_path):
		os.makedirs(out_path)
	all_classes = ['GM','WM','CSF','other'] # GM must be in front of WM
	tissue_classes = []
	for c in all_classes:
		if c in prior_map.keys():
			tissue_classes.append(c)
	all_classes = tissue_classes
	n_classes = len(all_classes)

	img_data = img.flatten()
	img_data_brain = img[brain_mask]
	img_max = np.percentile(img_data_brain,99)	
	# KDE 
	px_y = np.zeros((len(img_data_brain),n_classes))
	pts = np.arange(round(img_data_brain.max()*1.1))
	# Multi-modal density estimation for homo_classes
	homo_classes = ['WM','GM']
	peaks = {}
	for i in range(n_classes):
		C = all_classes[i]
		print(C)
		idx = sample_map[C]
		img_sample = img_data[idx]
		print('min prob of samples: '+str(prior_map[C].flatten()[idx].min()))
		if (C == 'other') and it==1:
			px_y[:,i] = 1/len(pts)
		else:
			if it==1:
				norm_path=out_path.split('probability')[-2]
			else:
				norm_path=None
			density,th,bw,mu_bounds = fitKDE(pts,n_classes,img,brain_mask,tissue_type=C,mr_type=con,img_sample=img_sample,bw_max=bw_max,bw_min=bw_min,norm_path=norm_path)
			sigma_l = round(bw*0.8)
			sigma_h = round(bw*15)
			sigma_lim = [sigma_l,sigma_h]

			plt.plot(pts,density,c='k')
			plt.axvline(x=th,c='r',ls='--')
			plt.text(th+2, density.max()/3, s=f'th={th}',fontsize=8)
			if C in homo_classes:
				maxima_intensity = argrelmax(density)[0]
				maxima_density = density[maxima_intensity]
				sort_idx = np.argsort(-maxima_density)
		
				# Keep the first two highest peaks
				highest = sort_idx[:2]
				maxima_density = maxima_density[highest]
				maxima_intensity = maxima_intensity[highest]
				# Discard the second highest if lower than 0.1*firstHighest
				keep = maxima_density>0.1*maxima_density.max()
				maxima_intensity = maxima_intensity[keep]				

				n_modal = len(maxima_intensity)
				if n_modal>1:
					print('found '+str(n_modal)+' peaks...using multimodal estimation')
					print('mu bounds: '+str(mu_bounds))
					try:
						Y,modal_intensity,sigma_ls,p = fitMultiModal(maxima_intensity,pts,density,bw,mu_bounds)

						for m in range(n_modal):
							modal_info = 'I={}, sigma={}, p={}'
							modal_info = modal_info.format(int(modal_intensity[m]),int(sigma_ls[m]),round(p[m],3))
							plt.text(modal_intensity[m], Y[:,m].max()/2, s=modal_info,fontsize=8)
						if p.min()<p_min[C][it]:
							print('Use higher probability peak')
							right_peak = np.argmax(p)
						else:
							sigma_inRange = (sigma_ls < sigma_h) & (sigma_ls > sigma_l)
							if  sigma_inRange.sum()==0:
								right_peak = -1
							elif sigma_inRange.sum()==1:
								right_peak = np.where(sigma_inRange)[0][0]
							else:
								diff = abs(modal_intensity[0]-modal_intensity[1])/modal_intensity.max()
								if 'dwi' in con:
									con_key='dwi'
								else:
									con_key=con

								if C == 'GM':
									if con == 'adc': # GM,WM contrast not good in adc, use higher prob
										right_peak = np.argmax(p)
									elif con in ['t1','t1post']:
										print('Use lower intensity peak')
										right_peak = np.argmin(modal_intensity)
									else:
										print('Use higher intensity peak')
										right_peak = np.argmax(modal_intensity)
								else: # WM
									WG_th_l = WM_GM_contrast[con_key][0]*peaks['GM']
									WG_th_h = WM_GM_contrast[con_key][1]*peaks['GM']
									plt.axvline(x=WG_th_l,ls='--')
									plt.axvline(x=WG_th_h,ls='--')
									WM_ex = (WG_th_l+WG_th_h)/2
									d2WM_ex = np.abs(modal_intensity-WM_ex)
									right_peak = np.argmin(d2WM_ex)

						if right_peak != -1:
							print('right_peak: '+str(modal_intensity[right_peak]))
							peaks[C] = modal_intensity[right_peak]
						else:
							print('No sigma in range. Did not change pdf')
							peaks[C] = maxima_intensity[0]
						for m in range(n_modal):
							if m == right_peak:
								c = 'b'
								density = Y[:,m]
							else:
								c = 'r'
							plt.plot(pts,Y[:,m],c=c)
					except ValueError:
						print('Multimodal estimation failed. Did not change pdf')
				else:
					peaks[C] = maxima_intensity[0]

			plt.title(C+' bw='+str(round(bw))+' sigma_lim: '+str(sigma_lim))
			plt.savefig(out_path+C+' density.png')
			plt.close()
			px_y[:,i] = density[np.round(img_data_brain).astype(int)]

	print(peaks)

	# Update posterior
	print('update posterior')
	px = np.sum(px_y,axis=1)
	posterior_map = {}
	sigma_x=brain_mask.shape[0]*0.005
	sigma_y=brain_mask.shape[1]*0.005
	sigma_z=brain_mask.shape[2]*0.005
	for i in range(n_classes):
		C = all_classes[i]
		prior = prior_map[C][brain_mask]
		posterior = np.zeros(brain_mask.shape)
		posterior[brain_mask] = prior*px_y[:,i]/px
		if smooth:
			posterior = gaussian_filter(posterior,[sigma_x,sigma_y,sigma_z]) 
		posterior[~brain_mask]=0
		posterior_map[C] = posterior

	posterior_map = normalizeProb(posterior_map,brain_mask)
	saveMaps(posterior_map,out_path,affine)

	return posterior_map

def tumorDetection(prior_map,brain_mask,input_images,n_iters,outpath,affine,n_jobs=1,bw_max=np.inf,bw_min=1,WM_GM_contrast=None,smooth=False):
	all_classes = list(prior_map.keys())
	n_classes = len(all_classes)
	contrasts = list(input_images.keys())
	n_channels = len(contrasts)
	posterior_map =  prior_map
	out_path = outpath+'posterior/'
	if not os.path.exists(out_path):
		os.makedirs(out_path)
	for it in range(1,n_iters+1):
		print('')
		print('iter: '+str(it))
		out_path = outpath+'posterior/iter '+str(it)+'/'
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		print('get tissue samples...')
		sample_map = getTissueSamples(posterior_map,brain_mask,input_images,n_jobs)
		print('Calculate posterior...')
		posterior_map_upadte = {}
		for C in all_classes:
			posterior_map_upadte[C]=np.zeros(brain_mask.shape)
		for con in contrasts:
			prob_map = updatePosterior(posterior_map,brain_mask,sample_map,input_images[con],affine,out_path+con+'/',it,bw_max,bw_min,WM_GM_contrast,smooth) 
			for C in all_classes:
				posterior_map_upadte[C] = posterior_map_upadte[C]+prob_map[C]/n_channels

		posterior_map = normalizeProb(posterior_map_upadte,brain_mask)
		saveMaps(posterior_map,out_path,affine)
	return posterior_map
