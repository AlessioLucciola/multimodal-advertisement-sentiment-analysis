from config import DATA_DIR
import os
import pickle
import numpy as np
import json
import h5py


data_segments_path = os.path.join(
    DATA_DIR, "GREX", '3_Physio', 'Transformed')

annotation_path = os.path.join(
    DATA_DIR, "GREX", '4_Annotation', 'Transformed')

# NOTE: Important keys here are: "filt_PPG" and "raw_PPG". Sampling rate is 100.
physio_trans_data_segments = pickle.load(
    open(os.path.join(data_segments_path, "physio_trans_data_segments.pickle"), "rb"))

# physio_trans_data_session = pickle.load(
#     open(os.path.join(data_segments_path, "physio_trans_data_session.pickle"), "rb"))

# NOTE: Important keys here are: 'ar_seg' and "vl_seg"
annotations = pickle.load(
    open(os.path.join(annotation_path, "ann_trans_data_segments.pickle"), "rb"))

print(f"Number of ar_seg are: {len(annotations['ar_seg'])}")
print(f"Number of vl_seg are: {len(annotations['vl_seg'])}")
print(
    f"filt_PPG shape {np.array(physio_trans_data_segments['filt_PPG']).shape}")

first_ppg = physio_trans_data_segments['filt_PPG'][0]

with open("first_ppg.json", "w") as f:
    json.dump(first_ppg, f, indent=4, default=list)
# with open("segments.json", "w") as f:
#     json.dump(physio_trans_data_segments, f, indent=4, default=str)

# with open("session.json", "w") as f:
#     json.dump(physio_trans_data_session, f, indent=4, default=str)

# with open("annotations.json", "w") as f:
#     json.dump(annotations, f, indent=4, default=str)
