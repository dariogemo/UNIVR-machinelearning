import optuna
import numpy as np
from keras.src.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score


def create_model(trial, input_dim):
    """Creates a model with two distinct sections as specified."""
    input_layer = Input(shape=(input_dim,))
    x = input_layer

    # Early layers (more neurons + dropout)
    num_early_layers = trial.suggest_int("num_early_layers", 1, 3)
    for i in range(num_early_layers):
        num_neurons = trial.suggest_int(f"early_neurons_{i}", 64, 128, step=8)
        x = Dense(num_neurons, activation=None, kernel_initializer="glorot_normal")(x)
        x = LeakyReLU()(x)
        x = Dropout(trial.suggest_float(f"dropout_{i}", 0.1, 0.3))(x)
        x = BatchNormalization()(x)

    # Later layers (fewer neurons, no dropout)
    num_later_layers = trial.suggest_int("num_later_layers", 1, 3)
    for i in range(num_later_layers):
        num_neurons = trial.suggest_int(f"later_neurons_{i}", 8, 64, step=8)
        x = Dense(num_neurons, activation=None, kernel_initializer="glorot_normal")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)  # Keeping batch norm for stability

    # Output layer
    output_classification = Dense(
        7,
        activation="softmax",
        kernel_initializer="glorot_normal",
        name="classification",
    )(x)

    model = Model(inputs=input_layer, outputs=output_classification)

    # Tune learning rate
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def objective(trial, X_train, y_train, X_val, y_val):
    """Objective function for Optuna hyperparameter tuning"""
    pruning = optuna.integration.TFKerasPruningCallback(trial, "val_accuracy")
    early_stop = EarlyStopping("val_accuracy", patience=5)

    model = create_model(trial, X_train.shape[1])

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        epochs=50,
        verbose=0,
        callbacks=[early_stop, pruning],
    )

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    return accuracy_score(y_val, y_pred)

