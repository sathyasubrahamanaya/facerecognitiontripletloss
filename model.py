# model.py
import torch
from facenet_pytorch import InceptionResnetV1

def load_embedding_model(model_path="resnet_face_triplet.pth", device="cpu"):
    """
    Loads the fine-tuned InceptionResnetV1 model for face embeddings.
    If the model_path is provided and loadable, it loads the weights;
    otherwise, it falls back to the default pretrained model.
    """
    # Load the pretrained model (using vggface2 weights by default)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Could not load model weights from {model_path}. Using default pretrained weights. Error: {e}")
    return model

def compute_embedding(image_tensor, model, device="cpu"):
    """
    Computes the face embedding for the given preprocessed image tensor.
    The image_tensor should be of shape (1, 3, H, W) and normalized to [-1,1].
    Returns a numpy array.
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.cpu().numpy()
