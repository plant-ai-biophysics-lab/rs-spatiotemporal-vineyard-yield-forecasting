import os
import numpy as np
import torch

#custom function 
from src.UNet import UNet2DConvLSTM_MS, UNet2DConvLSTM, UNet2DConvLSTM_S1
from src.datatools import data_weight_sampler, ReadData_V6, ReadData_V7, ReadData_S1, data_weight_sampler_eval
from src import utils 

#==============================================================================================================#
#==================================================Initialization =============================================#
#==============================================================================================================#
def eval(cultivar_list = None, patch_size = 80, patch_offset = 40, dropout = 0.3, batch_size = 64, spatial_resolution = 10, exp_name = None):
    # defult values 

    #==============================================================================================================#
    #================================================== Root Directories  =========================================#
    #==============================================================================================================#
    # the raw data directory
    if spatial_resolution == 1: 
        data_dir           = '/data2/hkaman/Livingston/data/1m/'
        bsize = 10
    elif spatial_resolution == 10:
        data_dir           = '/data2/hkaman/Livingston/data/10m/'
        bsize = 2
    if cultivar_list is None: 
        cultivar_list = ['MALVASIA_BIANCA', 'MUSCAT_OF_ALEXANDRIA', 'CABERNET_SAUVIGNON','SYMPHONY', 'MERLOT', 'CHARDONNAY', 'SYRAH', 'RIESLING']

    #exp_output_dir = '/data2/hkaman/Livingston/EXPs/10m/' + 'EXP_' + exp_name
    exp_output_dir = './inference/EXPs/' + 'EXP_' + exp_name

    best_model_name      = exp_output_dir + '/best_model' + exp_name + '.pth'
    train_df_name        = exp_output_dir + '/' + exp_name + '_train.csv'
    valid_df_name        = exp_output_dir + '/' + exp_name + '_valid.csv'
    test_df_name         = exp_output_dir + '/' + exp_name + '_test.csv'

    train_bc_df1_name    = exp_output_dir + '/' + exp_name + '_train_bc1.csv'
    train_bc_df2_name    = exp_output_dir + '/' + exp_name + '_train_bc2.csv'


    valid_bc_df1_name    = exp_output_dir + '/' + exp_name + '_valid_bc1.csv'
    valid_bc_df2_name    = exp_output_dir + '/' + exp_name + '_valid_bc2.csv'

    test_bc_df1_name      = exp_output_dir + '/' + exp_name + '_test_bc1.csv'
    test_bc_df2_name     = exp_output_dir + '/' + exp_name + '_test_bc2.csv'


    #==============================================================================================================#
    #================================================== csv file generator=========================================#
    #==============================================================================================================#
    #==============================================================================================================#
    #============================================      Data Weight Generation     =================================#
    #==============================================================================================================#
    train_sampler, val_sampler, test_sampler  = data_weight_sampler_eval(exp_output_dir)
    #==============================================================================================================#
    #============================================     Reading Data Senario 1      =================================#
    #==============================================================================================================#
   
    dataset_training = ReadData_S1(data_dir, exp_output_dir , category = 'train', window_size = patch_size)
    dataset_validate = ReadData_S1(data_dir, exp_output_dir , category = 'val',   window_size = patch_size)
    dataset_test     = ReadData_S1(data_dir, exp_output_dir , category = 'test',  window_size = patch_size)

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

    model = UNet2DConvLSTM_S1(in_channels = 4, out_channels =  1, num_filters = 16,
                                dropout = dropout, Emb_Channels = 0, 
                                batch_size = batch_size, botneck_size = bsize).to(device)

    #================================================= Training ============================================

    #loading the model:
    model.load_state_dict(torch.load(best_model_name))

    train_output_files = []
    valid_output_files = []
    test_output_files = []

    with torch.no_grad():
        
        model.eval()
        #================ Train===========================
        for batch, sample in enumerate(data_loader_training):
            
            X_batch_train       = sample['image'].to(device)
            y_batch_train       = sample['mask'].to(device)
            ID_batch_train      = sample['block']
            Cult_batch_train    = sample['cultivar']
            Xcoord_batch_train  = sample['X']
            ycoord_batch_train  = sample['Y']

            list_y_train_pred = model(X_batch_train)
            
            
            y_true_train = y_batch_train.detach().cpu().numpy()
            
            ytpw1 = list_y_train_pred[0].detach().cpu().numpy()
            ytpw2 = list_y_train_pred[1].detach().cpu().numpy()
            ytpw3 = list_y_train_pred[2].detach().cpu().numpy()
            ytpw4 = list_y_train_pred[3].detach().cpu().numpy()
            ytpw5 = list_y_train_pred[4].detach().cpu().numpy()
            ytpw6 = list_y_train_pred[5].detach().cpu().numpy()
            ytpw7 = list_y_train_pred[6].detach().cpu().numpy()
            ytpw8 = list_y_train_pred[7].detach().cpu().numpy()
            ytpw9 = list_y_train_pred[8].detach().cpu().numpy()
            ytpw10 = list_y_train_pred[9].detach().cpu().numpy()
            ytpw11 = list_y_train_pred[10].detach().cpu().numpy()
            ytpw12 = list_y_train_pred[11].detach().cpu().numpy()
            ytpw13 = list_y_train_pred[12].detach().cpu().numpy()
            ytpw14 = list_y_train_pred[13].detach().cpu().numpy()
            ytpw15 = list_y_train_pred[14].detach().cpu().numpy()


            this_batch_train = {"block": ID_batch_train, "cultivar": Cult_batch_train, "X": Xcoord_batch_train, "Y": ycoord_batch_train,
                                "ytrue": y_true_train, "ypred_w1": ytpw1, "ypred_w2": ytpw2,"ypred_w3": ytpw3,"ypred_w4": ytpw4,"ypred_w5": ytpw5,"ypred_w6": ytpw6,"ypred_w7": ytpw7,"ypred_w8": ytpw8,
                                "ypred_w9": ytpw9,"ypred_w10": ytpw10,"ypred_w11": ytpw11,"ypred_w12": ytpw12,"ypred_w13": ytpw13,"ypred_w14": ytpw14,"ypred_w15": ytpw15}
            
            train_output_files.append(this_batch_train)
  
        train_df = utils.ScenarioEvaluation2D(train_output_files)
        train_df.to_csv(train_df_name)

        train_block_names  = utils.npy_block_names(train_output_files)
        df1d_train, df2d_train = utils.time_series_eval_csv(train_output_files, train_block_names, patch_size)
        df1d_train.to_csv(train_bc_df1_name)
        df2d_train.to_csv(train_bc_df2_name)

        #================== Validaiton====================
        for batch, sample in enumerate(data_loader_validate):
            
            X_batch_val       = sample['image'].to(device)
            y_batch_val       = sample['mask'].to(device)
            ID_batch_val      = sample['block']
            Cult_batch_val    = sample['cultivar']
            Xcoord_batch_val  = sample['X']
            ycoord_batch_val  = sample['Y']
            list_y_val_pred = model(X_batch_val)
                
            y_true_val    = y_batch_val.detach().cpu().numpy()

            yvpw1 = list_y_val_pred[0].detach().cpu().numpy()
            yvpw2 = list_y_val_pred[1].detach().cpu().numpy()
            yvpw3 = list_y_val_pred[2].detach().cpu().numpy()
            yvpw4 = list_y_val_pred[3].detach().cpu().numpy()
            yvpw5 = list_y_val_pred[4].detach().cpu().numpy()
            yvpw6 = list_y_val_pred[5].detach().cpu().numpy()
            yvpw7 = list_y_val_pred[6].detach().cpu().numpy()
            yvpw8 = list_y_val_pred[7].detach().cpu().numpy()
            yvpw9 = list_y_val_pred[8].detach().cpu().numpy()
            yvpw10 = list_y_val_pred[9].detach().cpu().numpy()
            yvpw11 = list_y_val_pred[10].detach().cpu().numpy()
            yvpw12 = list_y_val_pred[11].detach().cpu().numpy()
            yvpw13 = list_y_val_pred[12].detach().cpu().numpy()
            yvpw14 = list_y_val_pred[13].detach().cpu().numpy()
            yvpw15 = list_y_val_pred[14].detach().cpu().numpy()
            
            

            this_batch_val = {"block": ID_batch_val, "cultivar": Cult_batch_val, "X": Xcoord_batch_val, "Y": ycoord_batch_val, 
                                "ytrue": y_true_val, "ypred_w1": yvpw1, "ypred_w2": yvpw2, "ypred_w3": yvpw3, "ypred_w4": yvpw4, "ypred_w5": yvpw5, "ypred_w6": yvpw6, "ypred_w7": yvpw7, "ypred_w8": yvpw8,
                                "ypred_w9": yvpw9, "ypred_w10": yvpw10, "ypred_w11": yvpw11, "ypred_w12": yvpw12, "ypred_w13": yvpw13, "ypred_w14": yvpw14, "ypred_w15": yvpw15} 

                
            valid_output_files.append(this_batch_val)


        valid_df = utils.ScenarioEvaluation2D(valid_output_files)
        valid_df.to_csv(valid_df_name)

        valid_block_names  = utils.npy_block_names(valid_output_files)
        df1d_valid, df2d_valid = utils.time_series_eval_csv(valid_output_files, valid_block_names, patch_size)
        df1d_valid.to_csv(valid_bc_df1_name)
        df2d_valid.to_csv(valid_bc_df2_name)
        #=================== Test ========================
        for batch, sample in enumerate(data_loader_test):
            
            X_batch_test       = sample['image'].to(device)
            y_batch_test       = sample['mask'].to(device)
            ID_batch_test      = sample['block']
            Cult_batch_test    = sample['cultivar']
            Xcoord_batch_test  = sample['X']
            ycoord_batch_test  = sample['Y']
        

            list_y_test_pred = model(X_batch_test)
            #list_y_test_pred, list_test_b = model(X_batch_test)
            y_true_test = y_batch_test.detach().cpu().numpy()
            
            ytepw1 = list_y_test_pred[0].detach().cpu().numpy()
            ytepw2 = list_y_test_pred[1].detach().cpu().numpy()
            ytepw3 = list_y_test_pred[2].detach().cpu().numpy()
            ytepw4 = list_y_test_pred[3].detach().cpu().numpy()
            ytepw5 = list_y_test_pred[4].detach().cpu().numpy()
            ytepw6 = list_y_test_pred[5].detach().cpu().numpy()
            ytepw7 = list_y_test_pred[6].detach().cpu().numpy()
            ytepw8 = list_y_test_pred[7].detach().cpu().numpy()
            ytepw9 = list_y_test_pred[8].detach().cpu().numpy()
            ytepw10 = list_y_test_pred[9].detach().cpu().numpy()
            ytepw11 = list_y_test_pred[10].detach().cpu().numpy()
            ytepw12 = list_y_test_pred[11].detach().cpu().numpy()
            ytepw13 = list_y_test_pred[12].detach().cpu().numpy()
            ytepw14 = list_y_test_pred[13].detach().cpu().numpy()
            ytepw15 = list_y_test_pred[14].detach().cpu().numpy()

            this_batch_test = {"block": ID_batch_test, "cultivar": Cult_batch_test, "X": Xcoord_batch_test, "Y": ycoord_batch_test, 
                            "ytrue": y_true_test, "ypred_w1": ytepw1, "ypred_w2": ytepw2, "ypred_w3": ytepw3, "ypred_w4": ytepw4, "ypred_w5": ytepw5, "ypred_w6": ytepw6, "ypred_w7": ytepw7, 
                            "ypred_w8": ytepw8, "ypred_w9": ytepw9, "ypred_w10": ytepw10, "ypred_w11": ytepw11, "ypred_w12": ytepw12, "ypred_w13": ytepw13, "ypred_w14": ytepw14, "ypred_w15": ytepw15}
            
            
            test_output_files.append(this_batch_test)
        #np.save(test_npy_name, test_output_files) 
        #print("Test Data is Saved!")
        test_df = utils.ScenarioEvaluation2D(test_output_files)
        test_df.to_csv(test_df_name)

        test_block_names  = utils.npy_block_names(test_output_files)
        df1d, df2d = utils.time_series_eval_csv(test_output_files, test_block_names, patch_size)
        df1d.to_csv(test_bc_df1_name)
        df2d.to_csv(test_bc_df2_name)


if __name__ == "__main__":
    
    eval(cultivar_list = None, patch_size = 16, 
    patch_offset = 2, dropout = 0.3, batch_size = 64,
    spatial_resolution = 10, exp_name = 'S1_UNetLSTM_10m_time')        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
       





