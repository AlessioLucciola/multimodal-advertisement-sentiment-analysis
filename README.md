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

You can download the needed data from this [Google Drive Link](https://drive.google.com/drive/folders/1BgkLk7GfHc8lLyqnabeT4jpEQQALClcQ)

Inside the `data` folder, there should be these elements:
- For the audio models, put these files in the `AUDIO` directory:
    -   `audio_metadata_ravdess.csv`: A file containing the (self-generated) metadata of the ravdess audio files;
    -   `audio_metadata_all.csv`: A file containing the (self-generated) metadata of the merged datasets audio files;
    -   `audio_ravdess_files`: A folder in which to put the ravdess audio files (downloadable from Google Drive);
    -   `audio_merged_datasets_files`: A folder in which to put the merged datasets audio files (downloadable from Google Drive).
- For the video models, put these files in the `VIDEO` directory:
    - `ravdess_frames_YxY_w_pixels.csv`: A file containing the (self-generated) metadata of the merged datasets video files where YxY can be 48x48 or 234x234;

All the files required for the audio and video model are zipped in the "AUDIO" and "VIDEO" folders in Google Drive.
