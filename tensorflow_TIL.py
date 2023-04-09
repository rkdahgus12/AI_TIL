import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


from tensorflow.keras import layers

with tf.device('/device:GPU:0'):
    # Load the Fashion MNIST dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize the data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define the model architecture
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=20)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy:", test_acc)
