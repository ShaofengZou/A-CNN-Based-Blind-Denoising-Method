import keras
from keras import backend as K 
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.layers import Input, GlobalAveragePooling2D

# Support 'mobilenet' or 'inceptionv2' or 'vgg16'
def get_classication_model(model_name='mobilenet', image_size=224, class_num=10):
    if model_name == 'mobilenet':
        model = get_mobilenet(image_size, class_num)
    elif model_name == 'inceptionv2':
        model = get_inception_v2(image_size, class_num)
    elif model_name == 'vgg16':
        model = get_vgg_16(image_size, class_num)
    return model


def get_mobilenet(image_size=224, class_num=10):
    base_model = MobileNet((image_size, image_size, 3), alpha=1, include_top=False, pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False

    x = Dropout(0.75)(base_model.output)
    x = Dense(class_num, activation='softmax')(x)

    model = Model(base_model.input, x)

    return model

def get_inception_v2(image_size=512, class_num=10):
    base_model = InceptionResNetV2(input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')

    for layer in base_model.layers:
        layer.trainable = False

    x = Dropout(0.75)(base_model.output)
    x = Dense(class_num, activation='softmax')(x)

    model = Model(base_model.input, x)

    return model

def get_vgg_16(image_size=512, class_num=10):
    input_tensor = Input(shape=(image_size, image_size, 3))
    base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False) 

    for layer in base_model.layers:
        layer.trainable = False

    # add global pooling
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.75)(x)
    x = Dense(class_num, activation='softmax')(x)

    model = Model(base_model.input, x)

    return model


def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)