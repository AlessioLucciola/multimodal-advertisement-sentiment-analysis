from shared.constants import video_models_list
from models.VideoResnetX import VideoResNetX
from models.VideoDenseNet121 import VideoDenseNet121
from models.VideoCustomCNN import VideoCustomCNN
from models.VideoViTPretrained import VideoViTPretrained

# Select model
def select_model(model_name, hidden_size, num_classes, dropout_p):
    if model_name == 'resnet18' or model_name == 'resnet34' or model_name == 'resnet50' or model_name == 'resnet101':
        model = VideoResNetX(model_name, hidden_size, num_classes, dropout_p)
    elif model_name == 'densenet121':
        model = VideoDenseNet121(hidden_size, num_classes, dropout_p)
    elif model_name == 'custom-cnn':
        model = VideoCustomCNN(num_classes, dropout_p)
    elif model_name == 'vit-pretrained':
        model = VideoViTPretrained(hidden_size, num_classes, dropout_p)
    else:
        raise ValueError(f'Invalid Model Name: Options {video_models_list}')
    
    return model

def split_video(video_path: str, frame_range: int = 100):
    """
    Given a video path and a frame range, it returns N slices of the video,
    each of them containing `frame_range` frames.
    """
    #TODO: implement this
    pass
