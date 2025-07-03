import numpy as np
import nibabel as nib
from scipy.stats import pearsonr
import os
import sys
import subprocess

def compute_voxelwise_correlation(brain_data, ref_vector, movie_name, ref_name, template_file):
    ref_vector = np.asarray(ref_vector).flatten()
    nx, ny, nz, nSubjects = brain_data.shape

    if nSubjects != len(ref_vector):
        raise ValueError("Reference vector length must match number of subjects.")

    prefix = f"{movie_name}.{ref_name}"
    corr_map = np.full((nx, ny, nz), np.nan, dtype=np.float32)
    pval_map = np.full((nx, ny, nz), np.nan, dtype=np.float32)
    masked_corr_map = np.full((nx, ny, nz), np.nan, dtype=np.float32)
    sig_mask = np.zeros((nx, ny, nz), dtype=np.float32)

    data_reshaped = brain_data.reshape(-1, nSubjects)
    print(f"data_reshaped.shape {data_reshaped.shape}")

    for v in range(data_reshaped.shape[0]):
        voxel_ts = data_reshaped[v, :]
        mask = ~np.isnan(voxel_ts) & ~np.isnan(ref_vector)
        if np.sum(mask) < 3:
            continue
        try:
            r, p = pearsonr(voxel_ts[mask], ref_vector[mask])
            corr_map.flat[v] = r
            pval_map.flat[v] = p
            if p < 0.1:
                masked_corr_map.flat[v] = r
                sig_mask.flat[v] = 1
        except Exception:
            continue

    print("✔️ Correlation range:", np.nanmin(corr_map), np.nanmax(corr_map))

    template = nib.load(template_file)
    affine = template.affine
    header = template.header.copy()

    outdir = os.path.join("./AfniFiles", movie_name)
    os.makedirs(outdir, exist_ok=True)

    matdir = os.path.join("./Matrixs", movie_name)
    os.makedirs(matdir, exist_ok=True)

    # Stack into 4D volume
    all_maps = np.stack([corr_map, pval_map, masked_corr_map, sig_mask], axis=-1)
    tmp_nii = os.path.join(outdir, f"{prefix}_4brick.nii.gz")
    afni_prefix = os.path.join(outdir, f"{prefix}_typicality")

    nib.save(nib.Nifti1Image(all_maps.astype(np.float32), affine=affine, header=header), tmp_nii)

    # Convert to AFNI BRIK/HEAD
    subprocess.run(["3dcopy", tmp_nii, f"{afni_prefix}+tlrc"], check=True)
    subprocess.run(["3drefit", "-space", "TLRC", f"{afni_prefix}+tlrc"], check=True)

    # Label sub-bricks
    subprocess.run([
        "3drefit",
        "-sublabel", "0", "corr",
        "-sublabel", "1", "pval",
        "-sublabel", "2", "masked_corr",
        "-sublabel", "3", "sig_mask",
        f"{afni_prefix}+tlrc"
    ], check=True)

    # Save .npy
    print("masked corr shape:",masked_corr_map.shape)
    np.save(os.path.join(matdir, f"{prefix}_corr.npy"), masked_corr_map)

    os.remove(tmp_nii)
    print(f"✅ AFNI file saved: {afni_prefix}+tlrc")
    print(f"✅ Numpy file saved: {matdir}/{prefix}_corr.npy")

# === Run from CLI
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python compute_voxelwise_correlation.py <brain_data.npy> <ref_vector.npy> <movie_name> <ref_name> [template_file]")
        sys.exit(1)

    brain_data = np.load(sys.argv[1])
    ref_vector = np.load(sys.argv[2])
    movie_name = sys.argv[3]
    ref_name = sys.argv[4]
    template_file = sys.argv[5] if len(sys.argv) > 5 else '/Volumes/Labs/ramot/michalwe/Master/fMRI_Analysis/Pyth_analysis_June25/Movie3Typ_corr.nii.gz'

    compute_voxelwise_correlation(brain_data, ref_vector, movie_name, ref_name, template_file)
