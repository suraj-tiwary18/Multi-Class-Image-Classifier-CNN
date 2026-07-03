# Data Path
train_path = "dataset/train"
test_path = "dataset/test"

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load Data
train_data = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

train = train_data.flow_from_directory(
    train_path, 
    target_size = (150,150), 
    batch_size = 32, 
    class_mode='categorical'
    )
test = test_data.flow_from_directory(
    test_path, 
    target_size = (150, 150), 
    batch_size=32, 
    class_mode = 'categorical'
    )

print('Train data indices')
print(train.class_indices)

print('Test data index')
print(test.class_indices)

# Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image


model = Sequential()

# First convolution Layer
model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3), 
    activation='relu',
    input_shape= (150, 150, 3)
))

# First Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution Layer
model.add(Conv2D(
    filters=64,
    activation='relu',
    kernel_size=(3, 3)
))

# Second Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolution Layer
model.add(Conv2D(
    filters=128,
    activation='relu', 
    kernel_size=(3, 3)
))

# Third Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fourth convolution Layer
model.add(Conv2D(
    filters=256,
    activation='relu', 
    kernel_size=(3, 3)
))

# Fourth Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
model.add(Flatten()) 

# Hidden layer
model.add(Dense(units=256, activation='relu'))

# Output layer
model.add(Dense(units=7, activation="softmax"))

# Model summary
model.summary()

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train, validation_data = test, epochs=10)

# Save Model
model.save("mult_object_classifier.keras")