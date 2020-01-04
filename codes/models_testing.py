import torch
import matplotlib.pyplot as plt
import pickle

x_file_name = 'generated_data_x_test_with_sliding_window_10.pickle'
y_file_name = 'generated_data_y_test.pickle'

model_file_name = "model_lstm_trained_on_generated_data_params:" \
                  " Run(lr=0.01, batch_size=20, num_workers=2, hidden_size=10, num_epochs=20, sequence_len=10).pth"

with open('../data/' + x_file_name, 'rb') as handle1:
    X_test = pickle.load(handle1)

with open('../data/' + y_file_name, 'rb') as handle2:
    y_test = pickle.load(handle2)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

directory = "../trained_models/"
path = directory + model_file_name
the_model = torch.load(path)

predicts = the_model(X_test)

# view raw_signal -last 1000 data points:
predicts = predicts.detach().numpy()
plt.plot(predicts, label = "predict")
y_test = y_test.numpy()
plt.plot(y_test, label = "True")
plt.title(model_file_name)
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.show()