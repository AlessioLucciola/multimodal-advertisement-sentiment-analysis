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

-   `RAVDESS` folder with:
    -   `ravdess.csv`: A file containing the (self-generated) metadata of the audio files;
    -   `files`: A folder in which to put the audio files (downloadable from Google Drive).