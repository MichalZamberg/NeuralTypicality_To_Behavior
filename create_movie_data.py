import os
import time
import warnings
import gzip
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path
import glob


def create_movie_data(movie, subjects):
    """
    Create movie data array from AFNI BRIK files for multiple subjects.
    Uses numpy memmap to handle large arrays efficiently.
    
    Parameters:
    -----------
    movie : str
        Movie identifier used in the file path
    subjects : list
        List of subject identifiers
        
    Returns:
    --------
    MovieArray : numpy.ndarray
        Concatenated array of all subject data
    """
    # Start total processing timer
    total_start = time.time()
    
    base_path = f'/Volumes/Labs/ramot/All_Data/NewPipline/data_20_ap_rest/SocCog/{movie}'  # CHANGE TO CORRECT PATH!!!!!!!!!!!
    expected_shape = None
    n_subjects = len(subjects)
    subject_shapes = []

    # First pass: find a valid file to determine expected shape
    print("Determining expected data shape...")
    for subject in subjects:
        subject_dir = Path(base_path) / subject / 'results'
        afni_files = glob.glob(str(subject_dir / f'errts.{subject}.SocCog.Movie*.tproject+tlrc.BRIK*'))  
        if afni_files:
            afni_file = afni_files[0]
            if afni_file.endswith('.gz'):
                unzipped_file = afni_file[:-3]
                if not os.path.exists(unzipped_file):
                    try:
                        with gzip.open(afni_file, 'rb') as f_in:
                            with open(unzipped_file, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    except Exception as e:
                        warnings.warn(f'Could not unzip file {afni_file}: {e}')
                        continue
                afni_file = unzipped_file
            try:
                img = nib.load(afni_file)
                expected_shape = img.get_fdata().shape
                print(f"Found valid file for {subject}, expected shape: {expected_shape}")
                break
            except Exception as e:
                warnings.warn(f'Error loading file {afni_file}: {e}')
                continue
    if expected_shape is None:
        raise ValueError('No valid files found to determine expected shape. Please check file paths and permissions.')

    # Prepare memmap array for output
    output_shape = expected_shape + (n_subjects,)
    os.makedirs(os.path.join('./Matrixs', movie), exist_ok=True)
    filename = os.path.join('./Matrixs', movie, f'{movie}.npy')
    MovieArray = np.lib.format.open_memmap(filename, mode='w+', dtype=np.float32, shape=output_shape)

    for i, subject in enumerate(subjects):
        subject_start = time.time()
        print(f'{subject}')
        subject_dir = Path(base_path) / subject / 'results'
        afni_files = glob.glob(str(subject_dir / f'errts.{subject}.SocCog.Movie*.tproject+tlrc.BRIK*'))
        if not afni_files:
            warnings.warn(f'File missing for {subject}, inserting NaNs.')
            subject_data = np.full(expected_shape, np.nan, dtype=np.float32)
            MovieArray[..., i] = subject_data
            continue
        afni_file = afni_files[0]
        if afni_file.endswith('.gz'):
            unzipped_file = afni_file[:-3]
            if os.path.exists(unzipped_file):
                print(f'Unzipped file already exists, using: {unzipped_file}')
            else:
                print(f'Unzipping {afni_file}...')
                try:
                    with gzip.open(afni_file, 'rb') as f_in:
                        with open(unzipped_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                except Exception as e:
                    warnings.warn(f'Could not unzip file, but continuing: {e}')
            afni_file = unzipped_file
        try:
            img = nib.load(afni_file)
            subject_data = img.get_fdata().astype(np.float32)
        except Exception as e:
            warnings.warn(f'Error loading file {afni_file}: {e}, inserting NaNs.')
            subject_data = np.full(expected_shape, np.nan, dtype=np.float32)
            MovieArray[..., i] = subject_data
            continue
        if subject_data.shape != expected_shape:
            print(f'Shape mismatch for {subject}: Expected {expected_shape}, got {subject_data.shape}')
            padded_data = np.full(expected_shape, np.nan, dtype=np.float32)
            dims = [min(s1, s2) for s1, s2 in zip(subject_data.shape, expected_shape)]
            slices = tuple(slice(0, d) for d in dims)
            padded_data[slices] = subject_data[slices]
            subject_data = padded_data
        print(f'Subject data shape: {subject_data.shape}')
        MovieArray[..., i] = subject_data
        subject_elapsed = time.time() - subject_start
        print(f'Time for subject {subject}: {subject_elapsed:.2f} seconds')

    print(f'All subject data loaded. Final shape: {MovieArray.shape}')
    print(f'Movie data saved successfully as a memory-mapped array at {filename}!')
    total_elapsed = time.time() - total_start
    print(f'Total processing time: {total_elapsed:.2f} seconds')
    return MovieArray


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python create_movie_data.py <movie_name> <subject_list>")
        print("Example: python create_movie_data.py Movie1 'sub001,sub002,sub003'")
        sys.exit(1)
    
    movie_name = sys.argv[1]
    subject_list = sys.argv[2].split(',')
    
    print(f"Processing movie: {movie_name}")
    print(f"Subjects: {subject_list}")
    
    try:
        movie_array = create_movie_data(movie_name, subject_list)
        print(f"Successfully created movie array with shape: {movie_array.shape}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
