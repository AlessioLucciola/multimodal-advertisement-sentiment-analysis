'''
file: predict.py
author: @vincit0re
brief: make the prediction on a given image
date: 2023-05-10
'''

from utils import *
argparse = argparse.ArgumentParser("Facial Emotion Detection")
argparse.add_argument('--img_path', type=str, default='img.png', required=True)
argparse.add_argument('--model_path', type=str,
                      default='model.pt', required=True)

args = argparse.parse_args()

# make predictions
def predict(img_path, model_path):
    # load the model
    model = get_model(7, device, model_name='resnet18')
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # load the image
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)

    # predict
    with torch.no_grad():
        images = img.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs.data, 1)
        print(predicted)
    return predicted


if __name__ == '__main__':
    predict(args.img_path, args.model_path)
