#! /bin/bash
#
#

INPUT_PATH=$1
OUTPUT_PATH=$2
echo "INPUT_PATH: ${INPUT_PATH}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

if [[ ! -d ${OUTPUT_PATH} ]]; then
	mkdir $OUTPUT_PATH
fi

read -a strarr <<< ${INPUT_PATH//"/"/" "}
PATIENT_ID=${strarr[-1]}
echo "Running damage detection on $PATIENT_ID"

img=$(find ${INPUT_PATH}/* -maxdepth 0 -type f -name "*t1.nii.gz")
echo "Input t1: $img"
FILE_PREFIX="$(basename -- $img)"
FILE_PREFIX=${FILE_PREFIX/_t1.nii.gz/""}
echo "FILE_PREFIX: ${FILE_PREFIX}"
echo "Resampling to 1x1x1"
ResampleImage 3 ${img} ${OUTPUT_PATH}/${FILE_PREFIX}_t1.nii.gz 1x1x1 0


echo "Head croping"
robustfov -i ${OUTPUT_PATH}/${FILE_PREFIX}_t1.nii.gz -r ${OUTPUT_PATH}/${FILE_PREFIX}_t1_head.nii.gz

echo "Coregister other inputs to T1"
for img in ${INPUT_PATH}/*.nii.gz
do
    if [[ "${img}" != *"t1.nii.gz" ]]; then
        echo "Register ${img} to T1"
        filename="$(basename -- $img)"
        outputStr=${OUTPUT_PATH}/${filename/.nii.gz/""}_coreg_
        ./antsRegistrationSyNQuick.sh -d 3 -f ${OUTPUT_PATH}/${FILE_PREFIX}_t1_head.nii.gz -m ${img} -o ${outputStr} -e 10 -t r >> ${OUTPUT_PATH}/coreg_log
    fi
done

echo "Performing N4 bias correction"
python N4.py ${OUTPUT_PATH}/${FILE_PREFIX}_t1_head.nii.gz ${OUTPUT_PATH}/${FILE_PREFIX}_t1_head_bc.nii.gz

echo "Brain extraction using BET"
bet ${OUTPUT_PATH}/${FILE_PREFIX}_t1_head_bc.nii.gz ${OUTPUT_PATH}/${FILE_PREFIX}_bet.nii.gz -m -f 0.5
python modifyBetMask.py ${OUTPUT_PATH}/${FILE_PREFIX}_bet_mask.nii.gz ${OUTPUT_PATH}/${FILE_PREFIX}_bet_mask_modified.nii.gz 2

echo "Affine register ICBM template"
ImageMath 3 ${OUTPUT_PATH}/${FILE_PREFIX}_ref_brain.nii.gz m ${OUTPUT_PATH}/${FILE_PREFIX}_t1_head_bc.nii.gz ${OUTPUT_PATH}/${FILE_PREFIX}_bet_mask_modified.nii.gz

fixed=${OUTPUT_PATH}/${FILE_PREFIX}_ref_brain.nii.gz
moving=ICBM_atlas/ICBM_Template_brain.nii.gz
outputStr="${OUTPUT_PATH}/${FILE_PREFIX}_ICBM2ref_Affine_"
./antsRegistrationSyNQuick.sh -d 3 -f ${fixed} -m ${moving} -o ${outputStr} -e 10 -t a > ${OUTPUT_PATH}/reg_ICBM_log

antsApplyTransforms --dimensionality 3 --input ICBM_atlas/ICBM_prob_white.nii.gz --reference-image ${fixed} --output ${OUTPUT_PATH}/${FILE_PREFIX}_WM_prior.nii.gz --interpolation Linear \
--transform ${OUTPUT_PATH}/${FILE_PREFIX}_ICBM2ref_Affine_0GenericAffine.mat
antsApplyTransforms --dimensionality 3 --input ICBM_atlas/ICBM_prob_gray.nii.gz --reference-image ${fixed} --output ${OUTPUT_PATH}/${FILE_PREFIX}_GM_prior.nii.gz --interpolation Linear \
--transform ${OUTPUT_PATH}/${FILE_PREFIX}_ICBM2ref_Affine_0GenericAffine.mat
antsApplyTransforms --dimensionality 3 --input ICBM_atlas/ICBM_cb_msk.nii.gz --reference-image ${fixed} --output ${OUTPUT_PATH}/${FILE_PREFIX}_cb_mask.nii.gz --interpolation Linear \
--transform ${OUTPUT_PATH}/${FILE_PREFIX}_ICBM2ref_Affine_0GenericAffine.mat
ThresholdImage 3 ${OUTPUT_PATH}/${FILE_PREFIX}_ICBM2ref_Affine_Warped.nii.gz ${OUTPUT_PATH}/${FILE_PREFIX}_brain_mask.nii.gz  1 3000 1 0

echo "Run outlier tumor segmentation"
python3 outlierSeg.py 3 ${OUTPUT_PATH} ${FILE_PREFIX}_ > ${OUTPUT_PATH}/seg_log

echo "Getting white and gray matter, removing CSF"
python get_wgm.py ${OUTPUT_PATH} ${FILE_PREFIX}_

echo "Register ICBM to patient for cerebellum segmentation"
inputImg=${OUTPUT_PATH}/${FILE_PREFIX}_wgm.nii.gz
template=ICBM_atlas/ICBM_gwm.nii.gz
outputStr=${OUTPUT_PATH}/${FILE_PREFIX}_temp_wgm_Syn_
./antsRegistrationSyNQuick.sh -d 3 -f ${inputImg} -m ${template} -o ${outputStr} -e 10 >> ${OUTPUT_PATH}/reg_ICBM_log

antsApplyTransforms --dimensionality 3 --input ICBM_atlas/ICBM_cb_msk.nii.gz --reference-image ${inputImg} --output ${OUTPUT_PATH}/${FILE_PREFIX}_temp_cb_msk.nii.gz --interpolation Linear \
--transform ${OUTPUT_PATH}/${FILE_PREFIX}_temp_wgm_Syn_1Warp.nii.gz \
--transform ${OUTPUT_PATH}/${FILE_PREFIX}_temp_wgm_Syn_0GenericAffine.mat

echo "Generating cerebellum mask"
python cb_mask.py ${OUTPUT_PATH} ${FILE_PREFIX}_ -1

echo "Running normalization"
./run_norm.sh . ${OUTPUT_PATH} ${FILE_PREFIX}_ 	15000

echo "Detecting damage"
python damageDetection.py ${OUTPUT_PATH} suit_atlas ${FILE_PREFIX}_ 0

# Delete temp files
rm ${OUTPUT_PATH}/*bet*
rm ${OUTPUT_PATH}/*gm*
rm ${OUTPUT_PATH}/*wm*
rm ${OUTPUT_PATH}/*.mat
rm ${OUTPUT_PATH}/*Warped.nii.gz
rm ${OUTPUT_PATH}/*prior.nii.gz
rm ${OUTPUT_PATH}/*temp*
rm ${OUTPUT_PATH}/*peduncle_bounding*
rm ${OUTPUT_PATH}/*ref_brain*
rm ${OUTPUT_PATH}/*stem*
rm ${OUTPUT_PATH}/*png
rm -r ${OUTPUT_PATH}/probability
