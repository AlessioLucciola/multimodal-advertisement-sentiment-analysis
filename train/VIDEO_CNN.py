import torch
from video_config import RANDOM_SEED, USE_WANDB, MODEL_NAME, BATCH_SIZE, LR, N_EPOCHS, METADATA_CSV, REG, NUM_CLASSES 
from dataloaders.FER_dataloader import FERDataloader
from models.VIDEO_CNN import get_model
from train.loops.video_train_loop import train_eval_loop
from utils.utils import set_seed, select_device
from utils.video_utils import plot_results, evaluate_model

def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    train_loader, val_loader, test_loader = FERDataloader(data_dir=METADATA_CSV,
                                           batch_size=BATCH_SIZE)
    
    model = get_model(NUM_CLASSES, device, model_name=MODEL_NAME).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=LR,
                                                    epochs=N_EPOCHS,
                                                    steps_per_epoch=len(train_loader))                                   
    criterion = torch.nn.CrossEntropyLoss()

    # TODO: use the next one
    params = {
        'model_name': MODEL_NAME,
        'lr': LR,
        'weight_decay': REG,
        'num_epochs': N_EPOCHS,
        'num_classes': NUM_CLASSES,
        'batch_size': BATCH_SIZE,
    }

    trained_model, history = train_eval_loop(model, 
                                         train_loader, 
                                         val_loader, 
                                         device, 
                                         params)

    # config = {
    #     "architecture": "VIDEO_CNN",
    #     "scope": "video_emotion_recognition",
    #     "learning_rate": LR,
    #     "epochs": N_EPOCHS,
    #     'reg': REG,
    #     'batch_size': BATCH_SIZE,
    #     "hidden_size": "",
    #     "num_classes": "",
    #     "dataset": "",
    #     "optimizer": "",
    #     "dropout_p": "",
    #     "use_wandb": USE_WANDB,
    # }

    # train_eval_loop(device=device,
    #                 train_loader=train_loader,
    #                 val_loader=val_loader,
    #                 model=model,
    #                 config=config,
    #                 optimizer=optimizer,
    #                 scheduler=scheduler,
    #                 criterion=criterion,
    #                 resume=False)
    
    # plot the results
    plot_results(history)

    # Test
    test_loss, test_acc = evaluate_model(
        trained_model, test_loader, criterion, device)

    print("-"*100)
    print(f"| Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f} |")
    print("-"*100)

if __name__ == "__main__":
    main()