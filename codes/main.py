import torch
import torch.utils.data as Data
import numpy as np
import math
import time
from collections import OrderedDict
from params_optimization import RunBuilder, RunManager
from simple_neural_net_models import RNN_Net, LSTM_Net
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
torch.manual_seed(0)


def normalize_data(data,one_feature = False):
    scaler = StandardScaler()
    if one_feature == True:
        scaler.fit(data.reshape(-1, 1))
    else:
        scaler.fit(data)
    return scaler

def sliding_windows(data, seq_length,one_feature = False):
    x = []
    y = []
    for i in range(data.shape[0]-seq_length-1):
        row_x = data[i:(i+seq_length)]
        if one_feature == True:
            row_y = data[i + seq_length]
        else:
            row_y = data[i+seq_length,0] #to predict only the next value in the first column

        x.append(row_x)
        y.append(row_y)

    return np.array(x), np.array(y)

def window_gen(data_name, data, sequence_len, train_ratio=0.7,one_feature = False):
    split_point = int(data.shape[0] * train_ratio)
    if one_feature:
        data_train = data[:split_point]
        data_test = data[split_point:]
        data_train = data_train.reshape(-1, 1)
        data_test = data_test.reshape(-1, 1)

    else:
        data_train = data[:split_point, :]
        data_test = data[split_point:, :]

    #normalize X:
    scaler = normalize_data(data_train, one_feature)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)


    X_train, y_train = sliding_windows(data_train, sequence_len)
    X_test, y_test = sliding_windows(data_test, sequence_len)


    #save test raw_signal for future performance checking
    import pickle
    x_file_name = data_name + f"x_test_with_sliding_window_{sequence_len}.pickle"
    y_file_name = data_name + "y_test" + ".pickle"
    path = "../data/"
    with open(path + x_file_name, 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path + y_file_name, 'wb') as handle:
        pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return torch.tensor(X_train, dtype=torch.float32), \
           torch.tensor(X_test, dtype=torch.float32), \
           torch.tensor(y_train, dtype=torch.float32), \
           torch.tensor(y_test, dtype=torch.float32)


# to read saved data:
import pickle
with open("../data/generated_data.pickel", 'rb') as handle:
    generated_data = pickle.load(handle)
signal = generated_data

#choosing the params for looping each possible combination:
params = OrderedDict(
    lr=[.01], #lr=[.01, .001],
    batch_size=[20,50],
    num_workers=[2],
    hidden_size=[10],
    num_epochs = [20],
    sequence_len = [10]
)

#change the names for saving results:
data_name = "generated_data_"
model_name = "_lstm_trained_on_" + data_name

m = RunManager()

for run in RunBuilder.get_runs(params):

    X_train, X_test, y_train, y_test = window_gen(data_name, signal, run.sequence_len,one_feature = True)
    torch_dataset_train = Data.TensorDataset(X_train, y_train)

    net = LSTM_Net(n_feature=1, n_output=1, hidden_dim=run.hidden_size, n_layers=1, sequence_len=run.sequence_len)
    #or check RNN rez:
    #net = RNN_Net(n_feature=1, n_output=1, hidden_dim=run.hidden_size, n_layers=1, sequence_len=run.sequence_len)

    train_loader = Data.DataLoader(dataset=torch_dataset_train, batch_size=run.batch_size, shuffle=True, num_workers=run.num_workers)
    optimizer = torch.optim.SGD(net.parameters(), lr=run.lr)
    loss_func = torch.nn.MSELoss()

    m.begin_run(run, net, train_loader)

    print(f"\n^^^^^ the hyper params: {run} ^^^^")
    train_initial_loss = loss_func(torch.squeeze(net(X_train)), torch.squeeze(y_train))
    print("loss before tarining: {}".format(train_initial_loss.item()))

    #start training:
    run_duration = 0
    for epoch in range(run.num_epochs):
        m.begin_epoch()
        for step, (batch_x, batch_y) in enumerate(train_loader):  # for each training step
            prediction = net(batch_x)  # input x and predict based on x
            loss = loss_func(torch.squeeze(prediction), torch.squeeze(batch_y))  # must be (1. nn output, 2. target)
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            m.track_loss(loss)
            # m.track_num_correct(prediction, batch_y) #use for classifications problems

        if epoch % 10 == 0 or epoch == (run.num_epochs - 1):
            train_loss = loss_func(torch.squeeze(net(X_train)), torch.squeeze(y_train))
            print(f"train loss for params {run}: {train_loss.item()}")

            test_loss = loss_func(torch.squeeze(net(X_test)), torch.squeeze(y_test))
            print(f"test loss for params {run}: {test_loss.item()}")

            print("**end epoch**")

            m.track_train_loss(train_loss)
            m.track_test_loss(test_loss)
        m.end_epoch()

    print(f"train time: {time.time() - m.run_start_time }" )

    #saving model:
    file_name = "model" + model_name + f"params: {run}.pth"
    directory = "../trained_models/"
    path = directory + file_name
    torch.save(net, path)
    the_model = torch.load(path)

    m.end_run()

#save results:
# m.save('../results/results ' + model_name)

#m.print_results()
