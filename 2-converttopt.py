import os
import nibabel as nib
import torch
import numpy as np

def convert_nii_to_pt(input_dir, output_dir):
    """
    Convert all .nii or .nii.gz files in a folder to .pt format
    while preserving voxel labels.

    Args:
        input_dir (str): Path to the folder containing .nii files.
        output_dir (str): Path to save converted .pt files.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output folder exists

    for filename in os.listdir(input_dir):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".nii", ".pt").replace(".nii.gz", ".pt"))

            try:
                # Load the NIfTI file
                nii_data = nib.load(file_path).get_fdata()

                # Convert to integer tensor (preserves labels)
                nii_tensor = torch.tensor(nii_data, dtype=torch.int64)

                # Save as .pt file
                torch.save(nii_tensor, output_path)

                print(f"Converted: {filename} → {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage:
input_folder = "nii_files"  # Replace with your input folder
output_folder = "pt_files"  # Replace with your output folder

convert_nii_to_pt(input_folder, output_folder)
