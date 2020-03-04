import tensorflow as tf

@tf.custom_gradient
def grad_reverse(x):
    grad = lambda dy: -dy * 0.1
    return x, grad

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)

class convnet(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(32, (5,5), (1,1), 'same')
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPool2D((3,3), 2, 'same')
        self.conv2 = tf.keras.layers.Conv2D(32, (3,3), (1,1), 'same')
        self.conv3 = tf.keras.layers.Conv2D(64, (3,3), (2,2), 'same')
        self.conv4 = tf.keras.layers.Conv2D(64, (3,3), (1,1), 'same')

        self.avgpool = tf.keras.layers.AvgPool2D((7,7), (1,1), 'valid')
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        feat_out = x
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)

        feat_low = x
        feat_low = self.avgpool(x)
        feat_low = self.flatten(x)
        y_low = self.fc(feat_low)

        return feat_out, y_low

class Predictor(tf.keras.Model):
    def __init__(self, num_classes=8):
        super().__init__()
        self.pred_conv1 = tf.keras.layers.Conv2D(32, (3,3), (1,1), 'same')
        self.pred_bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.pred_conv2 = tf.keras.layers.Conv2D(num_classes, (3,3), (1,1), 'same')
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.pred_conv1(x)
        x = self.pred_bn1(x)
        x = self.relu(x)
        x = self.pred_conv2(x)
        px = self.softmax(x)
        return x, px