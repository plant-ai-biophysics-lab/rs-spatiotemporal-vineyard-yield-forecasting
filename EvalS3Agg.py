import os
import numpy as np
import torch
import pandas as pd

#custom function 
from src.UNet import UNet2DConvLSTM
from src.datatools import data_weight_sampler, ReadData_V6, ReadData_V7, data_weight_sampler_eval, ReadData_V7_Time, ReadData_
from src import utils 

#==============================================================================================================#
#==================================================Initialization =============================================#
#==============================================================================================================#
def eval(cultivar_list = None, year_list = None, patch_size = 80, patch_offset = 40, scenario = None, 
            dropout = 0.3, batch_size = 64, 
            in_channel = 5, emb_channel = 4, 
            spatial_resolution = 10, 
            exp_name = None):


    if spatial_resolution == 1: 
        data_dir           = '/data2/hkaman/Livingston/data/1m/'
        bsize = 10
        exp_output_dir = '/data2/hkaman/Livingston/EXPs/1m/' + 'EXP_' + exp_name
    elif spatial_resolution == 10:
        data_dir           = '/data2/hkaman/Livingston/data/10m/'
        bsize = 2
        exp_output_dir = '/data2/hkaman/Livingston/EXPs/10m/' + 'EXP_' + exp_name



    if cultivar_list is None: 
        cultivar_list = ['MALVASIA_BIANCA', 'MUSCAT_OF_ALEXANDRIA', 'CABERNET_SAUVIGNON','SYMPHONY', 'MERLOT', 'CHARDONNAY', 'SYRAH', 'RIESLING']

    
    print(exp_output_dir)

    best_model_name   = exp_output_dir + '/best_model' + exp_name + '.pth'


    train_df_name_20  = exp_output_dir + '/' + exp_name + '_train_20m.csv'
    train_df_name_30  = exp_output_dir + '/' + exp_name + '_train_30m.csv'
    train_df_name_60  = exp_output_dir + '/' + exp_name + '_train_60m.csv'


    valid_df_name_20  = exp_output_dir + '/' + exp_name + '_valid_20m.csv'
    valid_df_name_30  = exp_output_dir + '/' + exp_name + '_valid_30m.csv'
    valid_df_name_60  = exp_output_dir + '/' + exp_name + '_valid_60m.csv'

    test_df_name_20   = exp_output_dir + '/' + exp_name + '_test_20m.csv'
    test_df_name_30   = exp_output_dir + '/' + exp_name + '_test_30m.csv'
    test_df_name_60   = exp_output_dir + '/' + exp_name + '_test_60m.csv'




    #==============================================================================================================#
    #============================================      Data Weight Generation     =================================#
    #==============================================================================================================#
    train_csv, val_csv, test_csv = utils.scenario_csv_generator(scenario = scenario, spatial_resolution = spatial_resolution, 
                                                        img_size = patch_size, offset = patch_offset, cultivar_list = cultivar_list, year_list = year_list)

    #==============================================================================================================#
    #============================================      Data Weight Generation     =================================#
    #==============================================================================================================#
    #train_sampler, val_sampler, test_sampler  = data_weight_sampler_eval(exp_output_dir)
    train_sampler, val_sampler, test_sampler  = data_weight_sampler(train_csv, val_csv, test_csv, exp_output_dir)
    #==============================================================================================================#
    #============================================     Reading Data Senario 1      =================================#
    #==============================================================================================================#
    dataset_training = ReadData_(data_dir, exp_output_dir, category = 'train', patch_size = patch_size, in_channels = in_channel, 
                                                                                spatial_resolution = spatial_resolution, run_status = 'valid')
    dataset_validate = ReadData_(data_dir, exp_output_dir, category = 'val',  patch_size = patch_size, in_channels = in_channel, 
                                                                                spatial_resolution = spatial_resolution, run_status = 'valid')
    dataset_test     = ReadData_(data_dir, exp_output_dir, category = 'test',  patch_size = patch_size, in_channels = in_channel, 
                                                                                spatial_resolution = spatial_resolution, run_status = 'eval')
    #==============================================================================================================#
    #=============================================      Data Loader               =================================#
    #==============================================================================================================#  
    # define training and validation data loaders
    data_loader_training = torch.utils.data.DataLoader(dataset_training, batch_size=batch_size, 
                                                    shuffle=False, sampler=train_sampler, num_workers=8) #, collate_fn=utils.collate_fn) # 
    data_loader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=batch_size, 
                                                    shuffle=False, sampler=val_sampler, num_workers=8) #, collate_fn=utils.collate_fn)
    data_loader_test     = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                                    shuffle=False, sampler=test_sampler, num_workers=8) 

    #================================================ Model Calling =========================================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet2DConvLSTM(in_channels = in_channel, out_channels = 1, num_filters = 16, 
                    dropout = dropout, Emb_Channels = emb_channel, batch_size = batch_size, botneck_size = bsize).to(device)

    #================================================= Training ============================================

    #loading the model:
    model.load_state_dict(torch.load(best_model_name))

    train_ytrue_20, train_ytrue_30, train_ytrue_60, train_ypred_20, train_ypred_30, train_ypred_60 = [], [], [], [], [], []
    valid_ytrue_20, valid_ytrue_30, valid_ytrue_60, valid_ypred_20, valid_ypred_30, valid_ypred_60 = [], [], [], [], [], []
    test_ytrue_20, test_ytrue_30, test_ytrue_60, test_ypred_20, test_ypred_30, test_ypred_60 = [], [], [], [], [], []


    train_df_20 = pd.DataFrame()
    train_df_30 = pd.DataFrame()
    train_df_60 = pd.DataFrame()

    valid_df_20 = pd.DataFrame()
    valid_df_30 = pd.DataFrame()
    valid_df_60 = pd.DataFrame()

    test_df_20 = pd.DataFrame()
    test_df_30 = pd.DataFrame()
    test_df_60 = pd.DataFrame()

    with torch.no_grad():
        
        model.eval()
        #================ Train===========================
        for batch, sample in enumerate(data_loader_training):
            
            X_batch_train       = sample['image'].to(device)
            y_batch_train       = sample['mask'].to(device)
            C_batch_train       = sample['EmbMatrix'].to(device)


            
            list_y_train_pred = model(X_batch_train, C_batch_train)
            
            y_true_train = y_batch_train.detach().cpu().numpy()
            batch_of_patch_ytrue_20m = utils.patch_resize(y_true_train, spatial_resolution = 20, status = 'true')
            train_ytrue_20.append(batch_of_patch_ytrue_20m)
            batch_of_patch_ytrue_30m = utils.patch_resize(y_true_train, spatial_resolution = 30, status = 'true')
            train_ytrue_30.append(batch_of_patch_ytrue_30m)
            batch_of_patch_ytrue_60m = utils.patch_resize(y_true_train, spatial_resolution = 60, status = 'true')
            train_ytrue_60.append(batch_of_patch_ytrue_60m)

            
            ytpw15 = list_y_train_pred[14].detach().cpu().numpy()

            batch_of_patch_ypred_20m = utils.patch_resize(ytpw15, spatial_resolution = 20, status = 'pred')
            train_ypred_20.append(batch_of_patch_ypred_20m)
            batch_of_patch_ypred_30m = utils.patch_resize(ytpw15, spatial_resolution = 30, status = 'pred')
            train_ypred_30.append(batch_of_patch_ypred_30m)
            batch_of_patch_ypred_60m = utils.patch_resize(ytpw15, spatial_resolution = 60, status = 'pred')
            train_ypred_60.append(batch_of_patch_ypred_60m)


        train_ytrue_20 = np.concatenate(train_ytrue_20)
        train_ytrue_30 = np.concatenate(train_ytrue_30)
        train_ytrue_60 = np.concatenate(train_ytrue_60)
        train_ypred_20 = np.concatenate(train_ypred_20)
        train_ypred_30 = np.concatenate(train_ypred_30)
        train_ypred_60 = np.concatenate(train_ypred_60)



        train_df_20['ytrue'] = train_ytrue_20
        train_df_20['ypred_w15'] = train_ypred_20
        train_df_20.to_csv(train_df_name_20)

        train_df_30['ytrue'] = train_ytrue_30
        train_df_30['ypred_w15'] = train_ypred_30
        train_df_30.to_csv(train_df_name_30)

        train_df_60['ytrue'] = train_ytrue_60
        train_df_60['ypred_w15'] = train_ypred_60
        train_df_60.to_csv(train_df_name_60)


        #================== Validaiton====================
        for batch, sample in enumerate(data_loader_validate):
            
            X_batch_val       = sample['image'].to(device)
            y_batch_val       = sample['mask'].to(device)
            C_batch_val       = sample['EmbMatrix'].to(device)


            list_y_val_pred = model(X_batch_val, C_batch_val)
                
            y_true_val      = y_batch_val.detach().cpu().numpy()

            v_batch_of_patch_ytrue_20m = utils.patch_resize(y_true_val, spatial_resolution = 20, status = 'true')
            valid_ytrue_20.append(v_batch_of_patch_ytrue_20m)
            v_batch_of_patch_ytrue_30m = utils.patch_resize(y_true_val, spatial_resolution = 30, status = 'true')
            valid_ytrue_30.append(v_batch_of_patch_ytrue_30m)
            v_batch_of_patch_ytrue_60m = utils.patch_resize(y_true_val, spatial_resolution = 60, status = 'true')
            valid_ytrue_60.append(v_batch_of_patch_ytrue_60m)

            yvpw15  = list_y_val_pred[14].detach().cpu().numpy()

            v_batch_of_patch_ypred_20m = utils.patch_resize(yvpw15, spatial_resolution = 20, status = 'pred')
            valid_ypred_20.append(v_batch_of_patch_ypred_20m)
            v_batch_of_patch_ypred_30m = utils.patch_resize(yvpw15, spatial_resolution = 30, status = 'pred')
            valid_ypred_30.append(v_batch_of_patch_ypred_30m)
            v_batch_of_patch_ypred_60m = utils.patch_resize(yvpw15, spatial_resolution = 60, status = 'pred')
            valid_ypred_60.append(v_batch_of_patch_ypred_60m)

        valid_ytrue_20 = np.concatenate(valid_ytrue_20)
        valid_ytrue_30 = np.concatenate(valid_ytrue_30)
        valid_ytrue_60 = np.concatenate(valid_ytrue_60)
        valid_ypred_20 = np.concatenate(valid_ypred_20)
        valid_ypred_30 = np.concatenate(valid_ypred_30)
        valid_ypred_60 = np.concatenate(valid_ypred_60)



        valid_df_20['ytrue'] = valid_ytrue_20
        valid_df_20['ypred_w15'] = valid_ypred_20
        valid_df_20.to_csv(valid_df_name_20)

        valid_df_30['ytrue'] = valid_ytrue_30
        valid_df_30['ypred_w15'] = valid_ypred_30
        valid_df_30.to_csv(valid_df_name_30)

        valid_df_60['ytrue'] = valid_ytrue_60
        valid_df_60['ypred_w15'] = valid_ypred_60
        valid_df_60.to_csv(valid_df_name_60)
            
            

        #=================== Test ========================
        for batch, sample in enumerate(data_loader_test):
            
            X_batch_test       = sample['image'].to(device)
            y_batch_test       = sample['mask'].to(device)
            C_batch_test       = sample['EmbMatrix'].to(device)

            list_y_test_pred = model(X_batch_test, C_batch_test)
            #list_y_test_pred, list_test_b = model(X_batch_test)
            y_true_test = y_batch_test.detach().cpu().numpy()

            t_batch_of_patch_ytrue_20m = utils.patch_resize(y_true_test, spatial_resolution = 20, status = 'true')
            test_ytrue_20.append(t_batch_of_patch_ytrue_20m)
            t_batch_of_patch_ytrue_30m = utils.patch_resize(y_true_test, spatial_resolution = 30, status = 'true')
            test_ytrue_30.append(t_batch_of_patch_ytrue_30m)
            t_batch_of_patch_ytrue_60m = utils.patch_resize(y_true_test, spatial_resolution = 60, status = 'true')
            test_ytrue_60.append(t_batch_of_patch_ytrue_60m)

            ytepw15 = list_y_test_pred[14].detach().cpu().numpy()


            t_batch_of_patch_ypred_20m = utils.patch_resize(ytepw15, spatial_resolution = 20, status = 'pred')
            test_ypred_20.append(t_batch_of_patch_ypred_20m)
            t_batch_of_patch_ypred_30m = utils.patch_resize(ytepw15, spatial_resolution = 30, status = 'pred')
            test_ypred_30.append(t_batch_of_patch_ypred_30m)
            t_batch_of_patch_ypred_60m = utils.patch_resize(ytepw15, spatial_resolution = 60, status = 'pred')
            test_ypred_60.append(t_batch_of_patch_ypred_60m)

        test_ytrue_20 = np.concatenate(test_ytrue_20)
        test_ytrue_30 = np.concatenate(test_ytrue_30)
        test_ytrue_60 = np.concatenate(test_ytrue_60)
        test_ypred_20 = np.concatenate(test_ypred_20)
        test_ypred_30 = np.concatenate(test_ypred_30)
        test_ypred_60 = np.concatenate(test_ypred_60)


        test_df_20['ytrue'] = test_ytrue_20
        test_df_20['ypred_w15'] = test_ypred_20
        test_df_20.to_csv(test_df_name_20)

        test_df_30['ytrue'] = test_ytrue_30
        test_df_30['ypred_w15'] = test_ypred_30
        test_df_30.to_csv(test_df_name_30)

        test_df_60['ytrue'] = test_ytrue_60
        test_df_60['ypred_w15'] = test_ypred_60
        test_df_60.to_csv(test_df_name_60)


if __name__ == "__main__":

    eval(cultivar_list = None, year_list= None, patch_size = 16, patch_offset = 2, 
    scenario = 3, dropout = 0.3, batch_size = 64, in_channel = 6, 
    emb_channel = 4, spatial_resolution = 10, exp_name = 'S3_UNetLSTM_10m_time')
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
       





