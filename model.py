import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import mlflow
mlflow.set_tracking_uri("http://134.209.232.89:5000/")
mlflow.tensorflow.autolog()
mlflow.set_experiment("MNIST")

with mlflow.start_run(run_name="test"):

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    input_shape = (28, 28, 1)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_train = x_train / 255.0
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_test = x_test / 255.0

    y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

    batch_size = 64
    num_classes = 10
    epochs = 10

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1)

    test_loss, test_acc = model.evaluate(x_test, y_test)

    # Predict the values from the testing dataset
    Y_pred = model.predict(x_test)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis = 1)
    # Convert testing observations to one hot vectors
    Y_true = np.argmax(y_test, axis = 1)
    # compute the confusion matrix
    confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt='g')

    print(history)

    train_losses=history.history['loss']
    train_accs=history.history['acc']
    val_losses=history.history['val_loss']
    val_accs=history.history['val_acc']

    tf.keras.models.save_model(model, "./model")


    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("epochs", epochs)
    #for train_loss in train_losses:
    #    mlflow.log_metric("train_loss", train_loss)
    #for train_acc in train_accs:
    #    mlflow.log_metric("train_accuracy", train_acc)
    #for val_loss in val_losses:
    #    mlflow.log_metric("val_loss", val_loss)
    #for val_acc in val_accs:
    #    mlflow.log_metric("val_accuracy", val_acc)

    mlflow.log_artifacts("./model")