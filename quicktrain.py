import os, time
from keras import backend as K
from keras.models import Model
import importlib
from keras.layers import Convolution2D, Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Activation, Flatten
import math
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from squeezenet_model.squeezenet import SqueezeNet 


# TensorFlow should use channels last
image_format='channels_last'
K.set_image_data_format( image_format )
# path to the model weights files.
save_model_name='from_squeezenet_imagenet.h5'
top_model_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models', save_model_name)

# dimensions of images. 
img_width, img_height = (227,227)

# nuumber of layers to freeze
nFreeze = 0

train_data_dir = '/data/shared/face_spoof/train'
validation_data_dir = '/data/shared/face_spoof/validation'
test_data_dir = '/data/shared/face_spoof/test'

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend


def add_top_layers(basemodel):
    #basemodel.summary()
    #list and cut layers here
    for k in range(len(basemodel.layers)):
        if k<56:
            basemodel.layers[k].trainable = False
        if ('input' in basemodel.layers[k].name):
            print('layer_{} name is {}, input shape={}'.format(k, basemodel.layers[k].name, basemodel.layers[k].input_shape))
        else:
            print('layer_{} name is {}, output={}'.format(k, basemodel.layers[k].name, basemodel.layers[k].output_shape ))
    # pop out layers we don't want
    number_to_pop=4
    for n in range(number_to_pop):
        print(basemodel.layers.pop())
    
     # the following works, however it is too big for the weights size   
        #x = Flatten()(basemodel.layers[len(basemodel.layers)-1].output)
        #x = Dense(256, activation='relu')(x)
        #x= Dropout(0.5)(x)
        #top_output= Dense(1, activation='sigmoid')(x)
    # use its original method but reduce class to 1
    x = Convolution2D(2, (1, 1), padding='valid', name='conv10')(basemodel.layers[len(basemodel.layers)-1].output)
    x = Activation('relu', name='relu_conv10')(x)
    x = Flatten()(x)
    #x = GlobalAveragePooling2D()(x)
    top_output= Dense(1, activation='sigmoid')(x)
    #top_output = Activation('softmax', name='loss')(x)

    # add the model on top of the convolutional base
    new_model=Model(inputs=basemodel.input, outputs=top_output)
    return new_model

def train_model(model):
    start_time = time.time()
    
    # freeze layers
    #for layer in model.layers[:nFreeze]:
    #    layer.trainable = False

    # compile model
    model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6),
              metrics=['accuracy'])
    
    # save model before training
    model.save_weights(top_model_weights_path)
    
    print( 'Model Compiled and saved')
    
    #model.load_weights(top_model_weights_path)
    #paper_weights_path =os.path.join(os.path.dirname(os.path.abspath(__file__)),'weights/REPLAY-ftweights18.h5')
    model.load_weights( top_model_weights_path )
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip = True,
            data_format=image_format,
            fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1./255, data_format=image_format)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=10,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=10,
            class_mode='binary')

    print ('\nFine-tuning top layers...\n')

    earlyStopping = callbacks.EarlyStopping(monitor='val_acc',
                                           patience=10, 
                                           verbose=0, mode='auto')

    #fit model
    batch_size = 10
    train_steps_per_epoth = math.floor(1766/batch_size)
    validation_steps_per_epoth = math.floor(1540/batch_size)
    model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoth, epochs=100, 
                        verbose=1, callbacks=[earlyStopping], validation_data=validation_generator, 
                        validation_steps=validation_steps_per_epoth, validation_freq=10, use_multiprocessing=True)


    model.save_weights(top_model_weights_path)
    
    print ('\nDone fine-tuning, have a nice day!')
    print("\nExecution time %s seconds" % (time.time() - start_time))


def Main():
    # change backend end to TF
    set_keras_backend('tensorflow')
    
    model_path='models'
    use_full_model = True    
    
    base_model = SqueezeNet(include_top=use_full_model, weights='imagenet', weights_path=model_path, full_model="squeezenet_weights_tf_dim_ordering_tf_kernels.h5", 
                        model_notop="squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5")

    fasmodel = add_top_layers(base_model)
    fasmodel.summary()
    #start to train the model
    train_model(fasmodel)
if __name__ == '__main__':
    
    Main()
    
    
