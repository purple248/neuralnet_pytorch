# Simple LSTM neural net and RNN with Pytorch with parameters optimization

This project contains the following codes files:

- data_generation:  creating a time series data with autocorrelation

- simple_neural_net_models: contains simple nets: dense net (not in use), RNN, LSTM

- params_optimization: a program to run through different parameter combination and tracking the loss for saving results

- main:
    Contains functions to prepare the data for training (normalization, generating windows)
    Choosing the parameters to loop for each parameters combination (one of them is the window len)
    Training the net and printing results during training
    Save the trained model

- models_testing: load the trained model and see results on the test data


