# multimodal-interaction-project

The work was carried out by:

- [Domiziano Scarcelli](https://github.com/DomizianoScarcelli)
- [Alessio Lucciola](https://github.com/AlessioLucciola)
- [Danilo Corsi](https://github.com/CorsiDanilo)


## Installation

We use Python 3.10.11 which is the last version supported by PyTorch. To create the environment using conda do

```
conda env create -f environment.yaml
conda activate mi_project
```

## Data

You can download the needed data from this [Google Drive Link](https://drive.google.com/drive/folders/1hN-QhdMj36LZ2GYIsmeDr20XiVLFrzZ1)

Inside the `data` folder, there should be these elements:
- For the audio models, put these files in the `AUDIO` directory:
    -   `audio_metadata_ravdess.csv`: A file containing the (self-generated) metadata of the ravdess audio files;
    -   `audio_metadata_all.csv`: A file containing the (self-generated) metadata of the merged datasets audio files;
    -   `audio_ravdess_files`: A folder in which to put the ravdess audio files (downloadable from Google Drive);
    -   `audio_merged_datasets_files`: A folder in which to put the merged datasets audio files (downloadable from Google Drive).
- For the video models, put these files in the `VIDEO` directory:
    - `RAVDESS_frames_files`: A folder containing the extracted frames from the video files (downloadable from Google Drive);
    - `RAVDESS_frames_files_black_background`: A folder containing the extracted frames from the video files with black background (downloadable from Google Drive);
    - `RAVDESS_metadata_original.csv`: A file containing the (self-generated) metadata of the video files (downloadable from Google Drive);
    - `RAVDESS_metadata_frames.csv`: A file containing the (self-generated) metadata of the frames (downloadable from Google Drive);
    - `RAVDESS_video_files`: A folder containing the original ravdess video files (downloadable from Google Drive);

All the files required for the audio and video model are zipped in the "AUDIO" and "VIDEO" folders in Google Drive.

## Demo

To run the demo, execute `streamlit run demo/init.py` or `python -m streamlit run demo/init.py`.
