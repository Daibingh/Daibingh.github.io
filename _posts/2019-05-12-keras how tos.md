---
layout: post
title:  "keras How tos"
categories: keras
tags: keras DL
author: hdb
comments: true
excerpt: "With keras, how to do something common?"
---

* content
{:toc}


## plot_model

- [FileNotFound error 解决方案](https://www.twblogs.net/a/5c179994bd9eee5e41844e1c/)

```py
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

## 记录训练损失

- use callback parameter in fit()
    ```py
    from keras.callbacks import CSVLogger
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```
- use return value by fit()
    ```py
    train_history = model.fit(X_train, Y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test))
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val_loss'])
    plt.show()
    ```

>https://stackoverflow.com/questions/39283358/keras-how-to-record-validation-loss

## keras 获取中间层输出和权重

### 获取权重

```py
weights = layer_model.get_weights()[0]
print(weights.shape)
```

### 获取中间层输出

```py
from keras import backend as K

# 方法 1
output_conv1 = K.function(inputs=[layer_input.input], outputs=[layer_conv1.output])
layer_output1 = output_conv1([[image1]])[0]
print(layer_output1.shape)

# 方法 2
output_conv2 = Model(inputs=layer_input.input, outputs=layer_conv2.output)
layer_output2 = output_conv2.predict(np.array([image1]))
print(layer_output2.shape)
```

## 自定义层

```py
from keras import backend as K
from keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

>https://keras.io/layers/writing-your-own-keras-layers/

## 使用回调函数

```py
class MyCallback(Callback):

    def __init__(self, train_feat=None, train_cap=None, test_feat=None, test_cap=None):
        self._best_score = .1
        self.train_feat = train_feat
        self.train_cap = train_cap
        self.test_feat = test_feat
        self.test_cap = test_cap

    def on_epoch_end(self, epoch, logs={}):
        train_bleu = bleu_score(self.train_feat, self.train_cap, encoder_model, decoder_model)
        test_bleu = bleu_score(self.test_feat, self.test_cap, encoder_model, decoder_model)
        print('train bleu: {} - test bleu: {}'.format(train_bleu, test_bleu))
        logs['train_bleu'] = train_bleu
        logs['test_bleu'] = test_bleu
        if test_bleu > self._best_score:
            model.save('./model/model_{}.mdl'.format(epoch))
            self._best_score = test_bleu
        return
```

## 重新初始化权重

```py
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
```

## 保存模型或权重

- model.save('xx.h5')
- model.save_weights('xx.h5')
- model = load_model('xx.h5')
- model.load_weights('xx.h5')

## 冻结层 

```py
base_model = ResNet50(include_top=False, input_shape=(224, 224, 3))
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(80, activation="softmax"))

for layer in base_model.layers:
    layer.trainable = False
model.load_weights('all_layers_freezed.h5')

for layer in base_model.layers[-26:]:
    layer.trainable = True
```

## 提前退出 fit()

```py
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
```

## 获取梯度

- https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras
- https://github.com/keras-team/keras/issues/2226#issuecomment-259004640

```py
def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return {s.name: t for s, t in zip(model.trainable_weights, output_grad)}


def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad
```

## 设置学习率衰减

```py
from keras.callbacks import LearningRateScheduler
def scheduler(epoch):
    if epoch == 5:
        model.lr.set_value(.02)
    return model.lr.get_value()

change_lr = LearningRateScheduler(scheduler)

model.fit(x_embed, y, nb_epoch=1, batch_size = batch_size, show_accuracy=True,
       callbacks=[chage_lr])
```
