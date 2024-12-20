<h1>Transfer Learning com Python</h1>
<p>Este projeto tem como objetivo implementar uma classificação binária para identificar Nina, uma gata, utilizando a técnica de <b>Transfer<br> Learning com o modélo pre treinado MobileNetV2.</b></p>
<br>
<h3>--Ferramentas Utilizadas--</h3>
<h4>Google_Colab, Drive</h4>
<br>
<h2>Código Fonte<h2>
<pre><code>
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
drive.mount('/content/drive')

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape = (128, 128, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1.0 / 255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Meu_Dataset/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
val_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Meu_Dataset/validation',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

loss, accuracy = model.evaluate(val_generator)
print(f"Loss: {loss}, Accuracy: {accuracy}")
</code></pre>
