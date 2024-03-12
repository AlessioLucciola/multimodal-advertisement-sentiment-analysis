from config import RAVDESS_CSV, RAVDESS_FILES, BATCH_SIZE
from dataloaders.RAVDESS_dataloader import get_test_dataloader, get_train_val_dataloaders
from datasets.RAVDESS_dataset import RAVDESSCustomDataset


def audio_model():
    ravdess_dataset = RAVDESSCustomDataset(csv_file=RAVDESS_CSV, files_dir=RAVDESS_FILES)
    train_loader, val_loader, val_df = get_train_val_dataloaders(ravdess_dataset, batch_size=BATCH_SIZE)
    test_loader = get_test_dataloader(val_df, batch_size=BATCH_SIZE)

    for batch in train_loader:
        tr_audio, tr_emotion, tr_emotion_intensity, tr_statement, tr_repetition, tr_actor = batch['audio'], batch['emotion'], batch['emotion_intensity'], batch['statement'], batch['repetition'], batch['actor']
        print(tr_audio, tr_emotion, tr_emotion_intensity, tr_statement, tr_repetition, tr_actor)

if __name__ == "__main__":
    audio_model()