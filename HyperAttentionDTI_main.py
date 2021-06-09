# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/7/05
@author: Qichang Zhao
"""
import random
import os
from model import AttentionDTI
from dataset import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from hyperparameter import hyperparameter
from pytorchtools import EarlyStopping  
import timeit
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score,precision_recall_curve, auc

def show_result(DATASET,lable,Accuracy_List,Precision_List,Recall_List,AUC_List,AUPR_List):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    print("The {} model's results:".format(lable))
    with open("./{}/results.txt".format(DATASET), 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))

def load_tensor(file_name, dtype):
    # return [dtype(d).to(hp.device) for d in np.load(file_name + '.npy', allow_pickle=True)]
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]

def test_precess(model,pbar,LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds, proteins, labels = data
            compounds = compounds.cuda()
            proteins = proteins.cuda()
            labels = labels.cuda()

            predicted_scores = model(compounds, proteins)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Reacll = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)  
    return Y, P, test_loss, Accuracy, Precision, Reacll, AUC, PRC

def test_model(dataset_load,save_path,DATASET, LOSS, dataset = "Train",lable = "best",save = True):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
        test_precess(model,test_pbar, LOSS)
    if save:
        with open(save_path + "/{}_{}_{}_prediction.txt".format(DATASET,dataset,lable), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(lable, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)
    return results,Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test


def get_kfold_data(i, datasets, k=5):
    
    fold_size = len(datasets) // k  

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]

    return trainset, validset

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
if __name__ == "__main__":
    """select seed"""
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    """init hyperparameters"""
    hp = hyperparameter()

    """Load preprocessed data."""
    DATASET = "KIBA"
    # DATASET = "DrugBank"
    # DATASET = "Davis"
    print("Train in " + DATASET)
    if DATASET == "DrugBank":
        weight_CE = None
        dir_input = ('./data/{}.txt'.format(DATASET))
        print("load data")
        with open(dir_input, "r") as f:
            train_data_list = f.read().strip().split('\n')
        print("load finished")
    elif DATASET == "Davis":
        weight_CE = torch.FloatTensor([0.3, 0.7]).cuda()
        dir_input = ('./data/{}.txt'.format(DATASET))
        print("load data")
        with open(dir_input, "r") as f:
            train_data_list = f.read().strip().split('\n')
        print("load finished")
    elif DATASET == "KIBA":
        weight_CE = torch.FloatTensor([0.2, 0.8]).cuda()
        dir_input = ('./data/{}.txt'.format(DATASET))
        print("load data")
        with open(dir_input, "r") as f:
            train_data_list = f.read().strip().split('\n')
        print("load finished")

    # random shuffle
    print("data shuffle")
    dataset = shuffle_dataset(train_data_list, SEED)
    K_Fold = 5

    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)
      
        train_dataset, test_dataset = get_kfold_data(i_fold, dataset)
        TVdataset = CustomDataSet(train_dataset)
        test_dataset = CustomDataSet(test_dataset)
        TVdataset_len = len(TVdataset)
        valid_size = int(0.2 * TVdataset_len)
        train_size = TVdataset_len - valid_size
        train_dataset, valid_dataset = torch.utils.data.random_split(TVdataset, [train_size, valid_size])
        train_dataset_load = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                        collate_fn=collate_fn)
        valid_dataset_load = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                        collate_fn=collate_fn)
        test_dataset_load = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                       collate_fn=collate_fn)

        """ create model"""
        model = AttentionDTI(hp).cuda()
        """weight initialize"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        """load trained model"""
        # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))

        # self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)
        
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)
        Loss = nn.CrossEntropyLoss(weight=weight_CE)
        # print(model)
        
        save_path = "./" + DATASET + "/{}".format(i_fold)
        note = ''
        writer = SummaryWriter(log_dir=save_path, comment=note)

        """Output files."""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path+'The_results_of_whole_dataset.txt'

        with open(file_results, 'w') as f:
            hp_attr = '\n'.join(['%s:%s' % item for item in hp.__dict__.items()])
            f.write(hp_attr + '\n')

        
        early_stopping = EarlyStopping(savepath = save_path,patience=hp.Patience, verbose=True, delta=0)
        # print("Before train,test the model:")
        # _,_,_,_,_,_ = test_model(test_dataset_load, save_path, DATASET, Loss, dataset="Test",lable="untrain",save=False)
        """Start training."""
        print('Training...')
        start = timeit.default_timer()

        for epoch in range(1, hp.Epoch + 1):
            trian_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(train_dataset_load)),
                total=len(train_dataset_load))
            """train"""
            train_losses_in_epoch = []
            model.train()
            for trian_i, train_data in trian_pbar:
                '''data preparation '''
                trian_compounds, trian_proteins, trian_labels = train_data
                trian_compounds = trian_compounds.cuda()
                trian_proteins = trian_proteins.cuda()
                trian_labels = trian_labels.cuda()
               
                optimizer.zero_grad()
               
                predicted_interaction = model(trian_compounds, trian_proteins)
                train_loss = Loss(predicted_interaction, trian_labels)
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss
            writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)
            # avg_train_losses.append(train_loss_a_epoch)

            """valid"""
            valid_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(valid_dataset_load)),
                total=len(valid_dataset_load))
            # loss_dev, AUC_dev, PRC_dev = valider.valid(valid_pbar,weight_CE)
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:
                    '''data preparation '''
                    valid_compounds, valid_proteins, valid_labels = valid_data

                    valid_compounds = valid_compounds.cuda()
                    valid_proteins = valid_proteins.cuda()
                    valid_labels = valid_labels.cuda()

                    valid_scores = model(valid_compounds, valid_proteins)
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]

                    valid_losses_in_epoch.append(valid_loss.item())
                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)
            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)  
            # avg_valid_loss.append(valid_loss)

            epoch_len = len(str(hp.Epoch))

            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Reacll: {Reacll_dev:.5f} ')

            writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
            writer.add_scalar('Valid AUC', AUC_dev, epoch)
            writer.add_scalar('Valid AUPR', PRC_dev, epoch)
            writer.add_scalar('Valid Accuracy', Accuracy_dev, epoch)
            writer.add_scalar('Valid Precision', Precision_dev, epoch)
            writer.add_scalar('Valid Reacll', Reacll_dev, epoch)
            writer.add_scalar('Learn Rate', optimizer.param_groups[0]['lr'], epoch)

            print(print_msg)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss_a_epoch, model, epoch)

        trainset_test_stable_results,_,_,_,_,_ = test_model(train_dataset_load, save_path, DATASET, Loss, dataset="Train", lable="stable")
        validset_test_stable_results,_,_,_,_,_ = test_model(valid_dataset_load, save_path, DATASET, Loss, dataset="Valid", lable="stable")
        testset_test_stable_results,Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
            test_model(test_dataset_load, save_path, DATASET, Loss, dataset="Test", lable="stable")
        AUC_List_stable.append(AUC_test)
        Accuracy_List_stable.append(Accuracy_test)
        AUPR_List_stable.append(PRC_test)
        Recall_List_stable.append(Recall_test)
        Precision_List_stable.append(Precision_test)
        with open(save_path + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')

    show_result(DATASET, "stable",
                Accuracy_List_stable, Precision_List_stable, Recall_List_stable,
                AUC_List_stable, AUPR_List_stable)



