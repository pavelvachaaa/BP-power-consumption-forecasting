import pandas as pd
from core import *
from NNModels import *
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import locale
import datetime
locale.setlocale(locale.LC_ALL,'cs_CZ')

import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style('white')


if __name__ == "__main__":
    wrapper = NNModels(scaler=MinMaxScaler(feature_range=(0, 1)))

    df: pd.DataFrame = load_london_dataset_household("./data/halfhourly_dataset/halfhourly_dataset/block_7.csv", "MAC004385", )
    df: pd.DataFrame = load_iris_dataset("./data/albistech_dataset/db3.json")

    df = wrapper.add_lags(df, Y_VALUE_NAME)
    df[Y_VALUE_NAME+"_diff"] = df[Y_VALUE_NAME].diff().fillna(0)
 
    df_back = df
    df = df[Y_VALUE_NAME].values.astype('float64')

    nn_dataset = wrapper.transform_for_lstm(df)
    train, test = wrapper.split_dataset(nn_dataset, train_size=0.7)

    # Rozdělíme si data na x a y
    X_train, Y_train = wrapper.to_sequence_for_lstm(train,2)
    X_test, Y_test = wrapper.to_sequence_for_lstm(test, 2)

    print("Shape (x-train): ",X_train.shape)
        # Shape (x-train):  (23942, 24)
        # 
    # Provedeme reshape X dat
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    print("Shape (x-train): ",X_train.shape)
    print("Shape (x-test): ",X_test.shape)
    #Shape (x-train):  (23942, 1, 24)
    # Shape (x-test):  (10247, 1, 24)
    model = wrapper.model_lstm_one(X_train.shape)
    # model = wrapper.model_cnn_lstm(X_train.shape)

    history = model.fit(X_train, Y_train, epochs=25, batch_size=64, validation_data=(X_test, Y_test),
            callbacks=[EarlyStopping(monitor='val_loss', patience=4)], verbose=1, shuffle=False)

    serialize_model(model, "lstm","beast2")
    loaded_model = deserialize_model("lstm", "beast2")

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = wrapper.scaler.inverse_transform(train_predict)
    Y_train = wrapper.scaler.inverse_transform([Y_train])
    test_predict = wrapper.scaler.inverse_transform(test_predict)
    Y_test = wrapper.scaler.inverse_transform([Y_test])


    A = Y_test[0][:96]
    F = test_predict[:,0][:96]

    evaluate_model(A,F)


 # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss']) 
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()      