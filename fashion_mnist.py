import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(f'Tensorflow Version:{tf.__version__}\n')

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(f"{train_images.shape}")
# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images/255
test_images = test_images/255

plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer = 'adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs = 10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print('\nTest Accuracy: ', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]

def plot_images(i, predictions_array, true_lable, img):
    true_lable, img = true_lable[i ], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predict_lable = np.argmax(predictions_array)
    if predict_lable == true_lable:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{}{:2.0f}% ({})".format(class_names[predict_lable],
                                        100*np.max(predictions_array),
                                        class_names[true_lable]),
                                        color = color)
    
def plot_value_array(i, predictions_array, true_lable):
    true_lable = true_lable[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim ([0,1])
    predict_lable = np.argmax(predictions_array)

    thisplot[predict_lable].set_color('red')
    thisplot[true_lable].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_images(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()