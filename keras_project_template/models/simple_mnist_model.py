from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Activation


class SimpleMnistModel(BaseModel):
    def __init__(self, config):
        super(SimpleMnistModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(28 * 28,)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(16))
        self.model.add(Activation('relu'))
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=self.config.optimizer,
            metrics=['acc'],
        )
