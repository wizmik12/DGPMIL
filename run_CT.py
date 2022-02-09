# libraries

import numpy as np
import pandas as pd
import itertools
import sys

import pickle
import os

import torch
torch.manual_seed(0)

from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.mlls import DeepApproximateMLL

from DGPMIL import DeepGPMIL, update_q_y
from mll_mil import VariationalELBO_MIL
from utils import instance_metrics_CT, bag_metrics_CT, bag_labels

# mlflow
import mlflow
mlruns_folder = 'mlruns'
experiment_name = 'CT_review_paper'
run_name = 'DGPMIL_8feat'

# grid of hyperparams
params_all = {"lr": [0.001],
"num_inducing": [10, 25, 50, 100, 150, 200, 300, 400, 500],
"dims": [[3], [3,3], [3,3,1]],
"num_epochs": [30],
"minibatch": [512],
"num_samples": [10]}

save_path = './probs/feat_8/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
# read test
def process_test(test_name, mean, std):
    test = pd.read_csv(DATA_PATH+test_name)
    test_x = test.iloc[:,:8].values # features
    test_x = torch.Tensor((test_x - mean) / std) # normalize
    if not 'CQ' in test_name: # RSNA has instance labels
        test_y = torch.Tensor(test["instance_label"].values)
    else:
        test_y = torch.Tensor(np.zeros(len(test_x)))
    bag_index_test = test["bag_name"].values.astype('str') # bag names
    y_bag_test_inst = test["bag_label"].values
    y_bag_test = bag_labels(bag_index_test, y_bag_test_inst) # bag labels

    return test_x, test_y, y_bag_test, bag_index_test

def group_metrics(list_metrics):
    # group the metrics of the 5 tests
    print('length metrics: ', len(list_metrics))
    concat_metrics =  pd.concat(list_metrics).groupby(level=0)
    print('concat_metrics: ', concat_metrics)
    mean_met, std_met = concat_metrics.mean().to_dict('list'), concat_metrics.std()
    std_met.columns = [x+'_std' for x in std_met.columns]
    std_met = std_met.to_dict('list')
    print('mean_metric: ', mean_met)
    for key, value in mean_met.items():
        mean_met[key]=value[0]
    for key, value in std_met.items():
        std_met[key]=value[0]
    print(mean_met)
    print(std_met)
    mlflow.log_metrics(mean_met)
    mlflow.log_metrics(std_met)
    return mean_met, std_met


keys, values = zip(*params_all.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
count = 1

# feature names
RSNA_train = ['RSNA_train_0.csv', 'RSNA_train_1.csv', 'RSNAadded_2.csv', 'RSNAadded_3.csv', 'RSNA_added_4.csv']
RSNA_test = ['CT_feature_final_8_0.csv', 'CT_feature_final_8_1.csv', 'RSNAadded_test_2.csv', 'RSNAadded_test_3.csv', 'RSNA_added_test_4.csv']
CQ500_test = ['CQ500_0.csv', 'CQ500_1.csv', 'CQ_added_2.csv', 'CQ_added_3.csv', 'CQ_added_4.csv']

for params in permutations_dicts:#range(len(permutations_dicts)):
    # List for saving metrics
    list_train_metrics_inst = []
    list_train_metrics_bag = []
    list_test_rsna_metrics_inst = []
    list_test_rsna_metrics_bag = []
    list_test_cq_metrics_inst = []
    list_test_cq_metrics_bag = []

    print('************** Combination ', count, '/', len(permutations_dicts))
    count = count +1
    mlflow.set_tracking_uri(mlruns_folder)
    experiment_id = mlflow.set_experiment(experiment_name)
    # print("exp_id", experiment_id.experiment_id)
    print(run_name)
    mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    #params = permutations_dicts[dict_param]
    params["hidden_layers"] = len(params["dims"])
    params["layers"] = len(params["dims"]) + 1

    mlflow.log_params(params)
    count = 0
    for train1, test1, test2 in zip(RSNA_train, RSNA_test, CQ500_test):

        # current fold
        print("train: ", train1)
        print("test_rsna: ", test1)
        print("test_cq500: ", test2)

        train1 = 'RSNA_train/' + train1
        test1 = 'RSNA_test/' + test1
        test2 = 'CQ500_test/' + test2

        DATA_PATH = 'data/'

        #Training data
        train = pd.read_csv(DATA_PATH+train1)
        train_x = train.iloc[:,:8].values
        mean, std = train_x.mean(0), train_x.std(0)
        train_x = torch.Tensor((train_x - mean) / std)
        train_y = torch.Tensor(np.arange(len(train_x)))
        true_train_y = torch.Tensor(train["instance_label"].values)
        y_bag_train_inst = train["bag_label"].values
        bag_index_train = train["bag_name"].values.astype('str')
        y_bag_train = bag_labels(bag_index_train, y_bag_train_inst)
        y_bag_train_inst = torch.Tensor(y_bag_train_inst)

        #Test data
        test_x_rsna, test_y_rsna, y_bag_test_rsna, bag_index_test_rsna = process_test(test1, mean, std)
        test_x_cq, test_y_cq, y_bag_test_cq, bag_index_test_cq = process_test(test2, mean, std)

        #Initialize q_y (instance labels)
        q_y = np.random.uniform(0, 0.1, size=len(train_x))
        q_y[y_bag_train_inst==0]=0.0
        q_y[y_bag_train_inst==1]=1.0
        q_y = torch.Tensor(np.vstack((1-q_y, q_y)).T)

        if torch.cuda.is_available():
            train_x, train_y = train_x.cuda(), train_y.cuda()
            test_x_rsna, test_y_rsna = test_x_rsna.cuda(), test_y_rsna.cuda()
            test_x_cq, test_y_cq = test_x_cq.cuda(), test_y_cq.cuda()
            true_train_y = true_train_y.cuda()
            q_y = q_y.cuda()
            y_bag_train_inst = y_bag_train_inst.cuda()
            print("Using GPU!")

        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=params["minibatch"], shuffle=True)
        train_loader_eval = DataLoader(train_dataset, batch_size=params["minibatch"], shuffle=False)
        test_dataset_rsna = TensorDataset(test_x_rsna, test_y_rsna)
        test_loader_rsna = DataLoader(test_dataset_rsna, batch_size=params["minibatch"])
        test_dataset_cq = TensorDataset(test_x_cq, test_y_cq)
        test_loader_cq = DataLoader(test_dataset_cq, batch_size=params["minibatch"])

        num_samples = params["num_samples"]
        num_epochs = params["num_epochs"]

        model = DeepGPMIL(train_x.shape[1], params["dims"].copy(), params["num_inducing"])

        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
        ], lr=params['lr'])
        mll = DeepApproximateMLL(VariationalELBO_MIL(model.likelihood, model, train_y.numel()))

        #epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
        epochs_iter = range(num_epochs)

        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            #minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=False)
            #for x_batch, y_batch in minibatch_iter:
            for x_batch, y_batch in train_loader:
                with gpytorch.settings.num_likelihood_samples(num_samples):
                    q_mb = q_y[y_batch.long()]
                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = -mll(output, q_mb)
                    loss.backward()
                    optimizer.step()

            q_y = update_q_y(q_y, model, train_loader_eval, bag_index_train, y_bag_train_inst)

            train_metrics, _ = instance_metrics_CT(model, train_loader_eval, true_train_y, 'train')
            test_rsna_metrics, _ = instance_metrics_CT(model, test_loader_rsna, test_y_rsna, 'test')

            print('Iter %d - Loss: %.3f' % (i + 1, loss.item()))
            print('Acc_train: ', train_metrics['acc_train'])
            print('Acc_test_RSNA: ', test_rsna_metrics['acc_test'])

        model.eval()

        with gpytorch.settings.num_likelihood_samples(100):
            list_train_metrics_inst.append(pd.DataFrame(instance_metrics_CT(model, train_loader_eval, true_train_y, 'train')[0], index = [0]))
            rsna_inst_metrics, prob_rsna_inst  = instance_metrics_CT(model, test_loader_rsna, test_y_rsna, 'test_RSNA')
            list_test_rsna_metrics_inst.append(pd.DataFrame(rsna_inst_metrics, index = [0]))

        test_rsna = pd.read_csv(DATA_PATH+test1)
        test_rsna['prob_dgpmil_inst'] = prob_rsna_inst
        test_rsna.to_csv(save_path + 'RSNA_DGPMIL'+ str(len(params["dims"]) + 1) + '_inducing_ ' + str(params["num_inducing"]) + '_' + str(count) + '.csv')

        with gpytorch.settings.num_likelihood_samples(100):
            list_train_metrics_bag.append(pd.DataFrame(bag_metrics_CT(model, train_loader_eval, y_bag_train, bag_index_train, 'train')[0], index = [0]))
            list_test_rsna_metrics_bag.append(pd.DataFrame(bag_metrics_CT(model, test_loader_rsna, y_bag_test_rsna, bag_index_test_rsna, 'test_rsna')[0], index = [0]))
            list_test_cq_metrics_bag.append(pd.DataFrame(bag_metrics_CT(model, test_loader_cq, y_bag_test_cq, bag_index_test_cq, 'test_cq')[0], index = [0]))

        count+=1

    
    # compute metrics
    mean_train_metrics_inst, std_train_metrics_inst = group_metrics(list_train_metrics_inst)
    mean_train_metrics_bag, std_train_metrics_bag = group_metrics(list_train_metrics_bag)

    mean_test_rsna_metrics_inst, std_test_rsna_metrics_inst = group_metrics(list_test_rsna_metrics_inst)
    mean_test_rsna_metrics_bag, std_test_rsna_metrics_bag = group_metrics(list_test_rsna_metrics_bag)

    mean_test_cq_metrics_bag, std_test_cq_metrics_bag = group_metrics(list_test_cq_metrics_bag)


    mlflow.end_run()
