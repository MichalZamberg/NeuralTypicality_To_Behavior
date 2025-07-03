import numpy as np
import sys
import os
import nibabel as nib
import subprocess

def category_mean(data1,data2,file_name,ref_file):

    brain_data=np.nanmean([brain_data1,brain_data2], axis=0)
    matdir = os.path.join("./Matrixs")
    os.makedirs(matdir, exist_ok=True)
    np.save(os.path.join(matdir, f"{file_name}_masked_corr.npy"), brain_data)
    
    template = nib.load(ref_file)
    affine = template.affine
    header = template.header.copy()

    outdir = os.path.join("./AfniFiles")
    os.makedirs(outdir, exist_ok=True)

    tmp_nii = os.path.join(outdir, f"{file_name}.nii.gz")
    afni_prefix = os.path.join(outdir, f"{file_name}_combined_map")
    nib.save(nib.Nifti1Image(brain_data.astype(np.float32), affine=affine, header=header), tmp_nii)

    # Convert to AFNI BRIK/HEAD
    subprocess.run(["3dcopy", tmp_nii, f"{afni_prefix}+tlrc"], check=True)
    subprocess.run(["3drefit", "-space", "TLRC", f"{afni_prefix}+tlrc"], check=True)
    os.remove(tmp_nii)




if __name__ == "__main__":
    brain_data1 = np.load(sys.argv[1])
    brain_data2 = np.load(sys.argv[2])
    prefix=sys.argv[3]
    ref = "/Volumes/Labs/ramot/All_Data/NewPipline/data_20_ap_rest/SocCog/Movie5/BD119/results/errts.BD119.SocCog.Movie5.tproject+tlrc.HEAD"

    category_mean(brain_data1,brain_data2,prefix,ref)





