#! /bin/bash
#
#
BASE_PATH=$1
OUTPUT_DIR=$2
prefix=$3
roi_th=$4
ATLAS_DIR=${BASE_PATH}/suit_atlas

ImageMath 3 ${OUTPUT_DIR}/${prefix}cb_wgm.nii.gz m ${OUTPUT_DIR}/${prefix}t1_head_bc.nii.gz ${OUTPUT_DIR}/${prefix}cb_mask.nii.gz


echo "Running affine registration: cerebellum to template"
inputImg=${OUTPUT_DIR}/${prefix}cb_wgm.nii.gz
template=${ATLAS_DIR}/SUIT.nii

outputStr="${OUTPUT_DIR}/${prefix}2temp_Affine_"
${BASE_PATH}/antsRegistrationSyNQuick.sh -d 3 -f ${template} -m ${inputImg} -o ${outputStr} -e 10 -t a > ${OUTPUT_DIR}/norm_log


echo "Apply transform to GM and WM"
antsApplyTransforms --dimensionality 3 --input ${OUTPUT_DIR}/${prefix}cb_gm.nii.gz --reference-image "${template}" --output ${OUTPUT_DIR}/${prefix}cb_gm_affine.nii.gz --interpolation Linear \
--transform ${OUTPUT_DIR}/${prefix}2temp_Affine_0GenericAffine.mat 
antsApplyTransforms --dimensionality 3 --input ${OUTPUT_DIR}/${prefix}cb_wm.nii.gz --reference-image "${template}" --output ${OUTPUT_DIR}/${prefix}cb_wm_affine.nii.gz --interpolation Linear \
--transform ${OUTPUT_DIR}/${prefix}2temp_Affine_0GenericAffine.mat
echo "Modify template if missing part is big"
python ${BASE_PATH}/rmMissFromTemp.py ${OUTPUT_DIR}/${prefix}2temp_Affine_Warped.nii.gz ${ATLAS_DIR}/SUIT.nii ${roi_th} 
ImageMath 3 ${OUTPUT_DIR}/temp_labeled.nii.gz m ${ATLAS_DIR}/suit_labeled.nii.gz ${OUTPUT_DIR}/temp_valid_roi.nii.gz

echo "Segment stem and label image"
python ${BASE_PATH}/segmentStem.py ${OUTPUT_DIR}/${prefix}2temp_Affine_Warped.nii.gz ${OUTPUT_DIR}/stem.nii.gz ${ATLAS_DIR}
python ${BASE_PATH}/combine_masks.py "${OUTPUT_DIR}/${prefix}cb_wm_affine.nii.gz" "${OUTPUT_DIR}/${prefix}cb_gm_affine.nii.gz" "${OUTPUT_DIR}/stem.nii.gz" "${OUTPUT_DIR}/${prefix}labeled.nii.gz" n

echo "Register labeled image"
template="${OUTPUT_DIR}/temp_labeled.nii.gz"
inputImg="${OUTPUT_DIR}/${prefix}labeled.nii.gz"
outputStr="${OUTPUT_DIR}/${prefix}2temp_labeled_Syn_"
${BASE_PATH}/antsRegistrationSyNQuick.sh -d 3 -f ${template} -m ${inputImg} -o ${outputStr} -e 10 >> ${OUTPUT_DIR}/norm_log

antsApplyTransforms --dimensionality 3 --input "${OUTPUT_DIR}/${prefix}cb_wgm.nii.gz" --reference-image ${template} --output ${OUTPUT_DIR}/${prefix}t1_norm.nii.gz --interpolation Linear \
--transform ${OUTPUT_DIR}/${prefix}2temp_labeled_Syn_1Warp.nii.gz \
--transform ${OUTPUT_DIR}/${prefix}2temp_labeled_Syn_0GenericAffine.mat \
--transform ${OUTPUT_DIR}/${prefix}2temp_Affine_0GenericAffine.mat

echo "Normalize other inputs"
for img in ${OUTPUT_DIR}/*coreg_Warped.nii.gz
do
    if [[ -f ${img} ]]; then
        echo "Apply cerebellar mask on ${img}"
        filename=${img/_coreg_Warped.nii.gz/""}_cb.nii.gz
        ImageMath 3 ${filename} m ${img} ${OUTPUT_DIR}/${prefix}cb_mask.nii.gz
        echo "Normalize ${filename}"
        antsApplyTransforms --dimensionality 3 --input "${filename}" --reference-image ${template} --output ${filename/"cb.nii.gz"/"norm.nii.gz"} --interpolation Linear \
        --transform ${OUTPUT_DIR}/${prefix}2temp_labeled_Syn_1Warp.nii.gz \
        --transform ${OUTPUT_DIR}/${prefix}2temp_labeled_Syn_0GenericAffine.mat \
        --transform ${OUTPUT_DIR}/${prefix}2temp_Affine_0GenericAffine.mat
    fi
done




