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

    df: pd.DataFrame = load_london_dataset_household("./data/halfhourly_dataset/halfhourly_dataset/block_0.csv", "MAC004387", )

    df = wrapper.add_lags(df, Y_VALUE_NAME)

    
    df_back = df
    df = df[Y_VALUE_NAME].values.astype('float64')


    nn_dataset = wrapper.transform_for_lstm(df)
    train, test = wrapper.split_dataset(nn_dataset, train_size=0.7)

    # Rozdělíme si data na x a y
    X_train, Y_train = wrapper.to_sequence_for_lstm(train,24)
    X_test, Y_test = wrapper.to_sequence_for_lstm(test, 24)

    print("Shape (x-train): ",X_train.shape)

    # Provedeme reshape X dat
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    print("Shape (x-train): ",X_train.shape)
    print("Shape (x-test): ",X_test.shape)

    model = wrapper.model_lstm_one(X_train.shape)
    # model = wrapper.model_cnn_lstm(X_train.shape)


    model.fit(X_train, Y_train, epochs=25, batch_size=50, validation_data=(X_test, Y_test),
            callbacks=[EarlyStopping(monitor='val_loss', patience=4)], verbose=1, shuffle=False)

    # code_name = "beast"
    # model.save(f"./out/models/lstm_model_{code_name}.h5", True, save_format='h5')
    serialize_model(model, "lstm","beast")
    loaded_model = deserialize_model("lstm", "beast")
    

    model.compile(loss="mse", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = wrapper.scaler.inverse_transform(train_predict)
    Y_train = wrapper.scaler.inverse_transform([Y_train])
    test_predict = wrapper.scaler.inverse_transform(test_predict)
    Y_test = wrapper.scaler.inverse_transform([Y_test])


    A = Y_test[0]
    F = test_predict[:,0]

    evaluate_model(A,F)
