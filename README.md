# Neural Activity To Behavior
# 🧠 fMRI-Behavior-Correlation
Overview

This project is a Python-based tool designed to correlate fMRI data (in AFNI BRIK/HEAD format) with behavioral scores. The goal is to streamline and automate the analysis of relationships between brain activation patterns and behavioral measures.

# ✨ Features

Supports AFNI .BRIK / .HEAD input files via nibabel.

Accepts behavioral data in .mat.

Handles subject matching based on filename conventions or metadata.

Computes voxel-wise correlation between brain data and behavioral scores.

Outputs correlation maps in AFNI-compatible NIfTI format.

Optionally corrects for multiple comparisons (e.g. FDR).

Visualization of brain-behavior correlations via plots or AFNI-compatible overlays.

# 📥 Input

fMRI Data
AFNI .BRIK / .HEAD pairs (per subject)

Must be 4D volumes (time series)

Behavioral Data
A npy file containing:
One or more columns with behavioral scores

# 📤 Output
Correlation map(s) in BRIK / HEAD format

# 🛠 Installation

Clone the repository:
git clone https://github.com/your-username/fMRI-Behavior-Correlation.git
cd fMRI-Behavior-Correlation

Install dependencies
We recommend using a virtual environment.

Dependencies include:
nibabel
numpy
pandas
scipy
matplotlib
nilearn (optional, for visualization and ROI tools)

# 🏃‍♀️ Running
The Codes Order:

create_movie_data(movie_name, subject_list)

compute_movie_typs(movie_name, subject_list, template_file, output_root)

compute_voxelwise_correlation(brain_data, ref_vector, movie_name, ref_name, template_file)

when you run localy don't forget to change the paths of the files in each of the codes.

# 📎 Notes

This tool assumes some familiarity with neuroimaging formats (AFNI, NIfTI).
If your data is in a different format (e.g. SPM, FSL), consider converting to NIfTI first.
Preprocessing steps (e.g. alignment, smoothing, masking) should ideally be done before using this tool.

# 📚 Attribution
This project was developed as part of the WIS Python programming course at the Weizmann Institute of Science.
The goal was to build a useful, real-world tool for research or personal use, extending beyond the course itself.

