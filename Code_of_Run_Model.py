import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('mult_object_classifier.keras')

img = image.load_img("PATOYS-Batman-Licensed-Electric-Ride-On-Bike-for-Kids-3-8-Years-Metallic-Black-PATOYS-4276363021_9912_1.jpg", target_size=(150,150))

img_array = image.img_to_array(img)

img_array = img_array/255.0

img_array = np.expand_dims(img_array, axis=0)

# Prediction
pred = model.predict(img_array)

classes = ["Bike", "Car", "Cat", "Dog", "Flowers", "Horse", "Human"]

print("Prediction : ", classes[np.argmax(pred)])