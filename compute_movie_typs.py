import os
import numpy as np
import nibabel as nib
import subprocess
import time
from pathlib import Path
import sys

def save_corr_as_afni(data_3d, label, output_dir, volume_shape, ref_nii):
    """
    Save a 3D numpy array as AFNI-aligned BRIK+HEAD file in +tlrc space.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_img = nib.load(ref_nii)
    affine = ref_img.affine
    header = ref_img.header.copy()

    # Safety check
    assert data_3d.shape == volume_shape, f"❌ Shape mismatch for {label}"

    nifti_path = output_dir / f"{label}.nii.gz"
    img = nib.Nifti1Image(data_3d.astype(np.float32), affine=affine, header=header)
    nib.save(img, nifti_path)

    afni_prefix = output_dir / label
    subprocess.run(["3dcopy", str(nifti_path), str(afni_prefix)], check=True)
    subprocess.run(["3drefit", "-view", "tlrc", f"{afni_prefix}+orig"], check=True)
    subprocess.run(["3drefit", "-space", "TLRC", str(nifti_path)], check=True)
    print(f"✅ Saved {label} as AFNI BRIK+tlrc and tagged as TLRC")

def compute_movie_typs(movie_name, subjects, template_nii, output_root="AfniFiles"):
    """
    Compute subject-level and group-average correlation maps for movie response data.
    Saves results in AFNI BRIK+HEAD format using Talairach-aligned reference.
    """
    start_time = time.time()

    # Load data
    movie_mat = np.load(os.path.join('Matrixs', movie_name, f'{movie_name}.npy'))  ## CHANGE TO CORRECT PATH!!!!!!!
    print(f"🎬 Loaded data shape: {movie_mat.shape}")
    X, Y, Z, TR, numSubs = movie_mat.shape
    volume_shape = (X, Y, Z)

    # Leave-one-out averaging
    Movie_avg = np.zeros_like(movie_mat)
    print("📊 Computing leave-one-out means...")
    for sub in range(numSubs):
        temp = np.delete(movie_mat, sub, axis=4)
        Movie_avg[..., sub] = np.nanmean(temp, axis=4)

    # Voxelwise correlations
    corr_all = np.zeros((X, Y, Z, numSubs))
    print("🔁 Computing voxelwise correlations...")
    for s in range(numSubs):
        x = movie_mat[..., s]
        y = Movie_avg[..., s]

        mx = np.nanmean(x, axis=3, keepdims=True)
        my = np.nanmean(y, axis=3, keepdims=True)

        x_c = x - mx
        y_c = y - my

        numerator = np.nansum(x_c * y_c, axis=3)
        std_x = np.sqrt(np.nansum(x_c ** 2, axis=3))
        std_y = np.sqrt(np.nansum(y_c ** 2, axis=3))
        denom = std_x * std_y

        r = numerator / denom
        r[denom == 0] = np.nan
        corr_all[..., s] = r

    # Save .npy file
    np.save(os.path.join('Matrixs', movie_name, f'{movie_name}_typs.npy'), corr_all)    ## CHANGE TO CORRECT PATH!!!!!!!
    print(f"💾 Saved 4D correlation matrix to Matrixs/{movie_name}_typs.npy")

    # Create output folder
    output_dir = Path(output_root) / movie_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save group-average as AFNI file
    group_avg = np.nanmean(corr_all, axis=3)
    save_corr_as_afni(group_avg, f"{movie_name}_typ_avg", output_dir, volume_shape, template_nii)

    # Save each subject’s AFNI file
    for s in range(numSubs):
        subj_corr = corr_all[..., s]
        subj_label = f"{movie_name}_typ_{subjects[s]}"
        save_corr_as_afni(subj_corr, subj_label, output_dir, volume_shape, template_nii)

    print(f"⏱️ Total processing time: {time.time() - start_time:.2f} seconds")
    return corr_all
