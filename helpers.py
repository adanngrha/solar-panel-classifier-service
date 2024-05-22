import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

model = None

def load_model():
  model = tf.keras.models.load_model('./best_model_MobileNetV2.h5')
  return model

def predict_image(image: Image.Image):
  global model
  if model is None:
    model = load_model()
    
  image = np.asarray(image.resize((224, 224)))[..., :3]
  image = np.expand_dims(image, 0)
  image = image / 127.5 - 1.0
  
  class_probabilities = model.predict(image) 

  class_names = ['clean', 'dusty', 'bird_drop', 'electrical_damage', 'physical_damage']
  predicted_classes_index = np.argmax(class_probabilities)
  predicted = class_names[predicted_classes_index]
        
  return predicted
    
def read_imagefile(file) -> Image.Image:
  image = Image.open(BytesIO(file))
  return image