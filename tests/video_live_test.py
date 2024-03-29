import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from config import MODEL_NAME, NUM_CLASSES
from utils.utils import select_device
from shared.constants import FER_emotion_mapping

from models.VIDEO.densenet121 import DenseNet121
from models.VIDEO.inception_v3 import InceptionV3
from models.VIDEO.resnetx_cnn import ResNetX
from models.VIDEO.custom_cnn import CustomCNN

def load_trained_model(model_path):
    device = select_device()

    # load the model
    if MODEL_NAME == 'resnet18' or MODEL_NAME == 'resnet34' or MODEL_NAME == 'resnet50' or MODEL_NAME == 'resnet101':
        model = ResNetX(MODEL_NAME, NUM_CLASSES)
    elif MODEL_NAME == 'dense121':
        model = DenseNet121(NUM_CLASSES)
    elif MODEL_NAME == 'inception_v3':
        model = InceptionV3(NUM_CLASSES)
    elif MODEL_NAME == 'custom_cnn':
        model = CustomCNN(NUM_CLASSES)
    else:
        raise ValueError('Invalid Model Name: Options [resnet18, resnet34, resnet50, resnet101, dense121, inception_v3, custom_cnn]')

    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

def FER_live_cam():
    model = load_trained_model('./checkpoints/video/'+ MODEL_NAME + '_best.pt') # change this to the path of the trained model

    val_transform = transforms.Compose([
        transforms.ToTensor()])

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier('./models/haarcascade/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
            X = resize_frame/256
            X = Image.fromarray((X))
            X = val_transform(X).unsqueeze(0)
            with torch.no_grad():
                model.eval()
                log_ps = model.cpu()(X)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                pred = FER_emotion_mapping[int(top_class.numpy())]
            cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    FER_live_cam()
