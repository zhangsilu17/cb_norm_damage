import SimpleITK as sitk
import sys

file_name = sys.argv[1]
output_name = sys.argv[2]
image=sitk.ReadImage(file_name)
inputImage = sitk.Cast(image, sitk.sitkFloat32)
corrector = sitk.N4BiasFieldCorrectionImageFilter()
outputImage = corrector.Execute(inputImage)
outputImage = sitk.Cast(outputImage,sitk.sitkUInt16)
sitk.WriteImage(outputImage, output_name)