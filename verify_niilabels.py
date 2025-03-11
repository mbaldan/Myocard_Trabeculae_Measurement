import nibabel as nib
import numpy as np

# Load the labeled segmentation file
seg_path = "heart_segmentation.nii.gz"
seg_img = nib.load(seg_path)

# Convert to NumPy array
seg_data = seg_img.get_fdata()

# Check unique voxel labels
unique_labels = np.unique(seg_data)
print("Unique Labels in Segmentation File:", unique_labels)
