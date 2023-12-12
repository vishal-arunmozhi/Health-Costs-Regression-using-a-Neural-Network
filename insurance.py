import sys

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.regularizers import l2

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TEST_SIZE = 0.2

def main():
    # check command line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage python insurance.py data")

    # load data from the spreadsheet and split it into train and test sets
    features, labels = load_data(sys.argv[1])

    # splitting dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=42
    )

    # create the model
    model = create_model()

    checkpoint_callback = ModelCheckpoint(filepath='best_model_weights.h5',
                                          monitor='loss',
                                          verbose=0,
                                          save_best_only=True,
                                          save_weights_only=True)

    # training the model
    history = model.fit(X_train, y_train, epochs=300, verbose=2, batch_size=32, callbacks=[checkpoint_callback], shuffle=True)

    plot_loss(history)

    # loading the model with the best weights
    model.load_weights('best_model_weights.h5')
    print("\nModel with the best weights loaded. Evaluating it on test data.")

    # evaluating the model
    print("\nModel Evaluation:")
    result = model.evaluate(X_test, y_test)
    print("\nMAE on test data: ", result)

# Function to take in the file name, retrieve and return the features and labels
def load_data(filename):
    dataset = pd.read_csv(filename)

    # converting categorical data to numerical data.
    dataset['sex'].replace({'male': 0, 'female': 1}, inplace=True)
    dataset['smoker'].replace({'no':0, 'yes': 1}, inplace=True)

    # one hot encoding for the 'region' column to make sure the model doesn't assume any ordinal relationsip between the categories
    dataset = pd.get_dummies(dataset, columns=['region'], prefix=['region'], dtype=int)

    # performing z-score normalization on necessary columns
    scaler = StandardScaler()
    dataset['age'] = scaler.fit_transform(dataset[['age']])
    dataset['bmi'] = scaler.fit_transform(dataset[['bmi']])
    dataset['children'] = scaler.fit_transform(dataset[['children']])

    # removing the 'expenses' column from the dataset and assigning it as labels
    labels = dataset.pop('expenses')

    features = dataset

    return features, labels

# Function to create the model architecture and return the compiled model
def create_model():
    layers = [
        # layer 1
        Dense(units=64, input_shape=(9,), activation='relu'),

        # layer 2
        Dense(units=64, activation='relu'),

        # layer 5 - output layer
        Dense(1)
    ]

    # using sequential model from keras to perfrom linear regression
    model = Sequential(layers)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='mae')

    return model

# Hyptertuning was performed using K-Fold Cross Validation
def cross_validation(features, labels):
    # converting pandas dataframe into numpy arrays
    features = features.values
    labels = labels.values

    # k-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    train_mae_scores = []
    test_mae_scores = []
    
    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        model = create_model()

        history = model.fit(X_train, y_train, 
                  epochs=400, verbose=0,
                  batch_size=32)
        train_mae = history.history['loss']

        y_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)

        train_mae_scores.append(np.mean(train_mae[-5:])) # for every iteration calculating the mean of the last 5 mae values during convergence
        test_mae_scores.append(test_mae) # comparing the train and test mae scores would tell us if the model is overfitting
        
    # Calculate the mean and standard deviation of the MAE scores
    mean_train_mae = np.mean(train_mae_scores)
    mean_test_mae = np.mean(test_mae_scores)
    std_mae = np.std(test_mae_scores)

    # formatting train and test mae scores to round them off to 3 decimal places
    train_mae_scores = [f'{mae:.3f}' for mae in train_mae_scores]
    test_mae_scores = [f'{mae:.3f}' for mae in test_mae_scores]

    # printing the results
    print(f'Train MAE scores: {train_mae_scores}')
    print(f'Test MAE scores: {test_mae_scores}')
    print(f'Mean Train MAE score: {mean_train_mae:.3f}')
    print(f'Mean Test MAE score: {mean_test_mae:.3f}')
    print(f'Standard Deviation of MAE: {std_mae:.3f}')

# Function to plot training losss
def plot_loss(history):
    training_loss = history.history['loss']

    epochs = list(range(1, len(training_loss) + 1))

    plt.figure(figsize=(13, 15))
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

