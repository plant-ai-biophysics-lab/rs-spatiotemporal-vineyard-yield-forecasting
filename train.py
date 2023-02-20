import os
import torch
import torch.nn as nn
from src.UNetConvLSTM import UNet2DConvLSTM
from src.DataLoader import return_cost_sensitive_weight_sampler, dataloader_, pixel_hold_out_dataloader
from src import utils, ModelEngine, Inference
import time
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
#==============================================================================================================#
#==================================================Initialization =============================================#
#==============================================================================================================#
def train(scenario: str, 
            spatial_resolution: int, 
            patch_size: int, 
            patch_offset: int,  
            cultivar_list: list,  
            year_list: list, 
            dropout: int, 
            batch_size: int, 
            learning_rate: float, 
            weight_decay: float,
            in_channel: int, 
            emb_channel: int, 
            loss_stop_tolerance: int, 
            epochs: int, 
            exp_name: str):

    if spatial_resolution == 1: 
        data_dir           = '/data2/hkaman/Livingston/data/1m/'
        botneck_size = 10
        exp_output_dir = '/data2/hkaman/Livingston/EXPs/1m/' + 'EXP_' + exp_name

    elif spatial_resolution == 10:
        data_dir           = '/data2/hkaman/Livingston/data/10m/'
        botneck_size = 2
        exp_output_dir = '/data2/hkaman/Livingston/EXPs/10m/' + 'EXP_' + exp_name

    
    isExist  = os.path.isdir(exp_output_dir)

    if not isExist:
        os.makedirs(exp_output_dir)
        os.makedirs(exp_output_dir + '/coords')
        os.makedirs(exp_output_dir + '/loss')
        os.makedirs(exp_output_dir + '/RVs')

    best_model_name      = exp_output_dir + '/best_model_' + exp_name + '.pth'
    loss_fig_name        = exp_output_dir + '/loss/loss_'  + exp_name + '.png'
    loss_df_name         = exp_output_dir + '/loss/loss_'  + exp_name + '.csv' 
    loss_weekly_fig_name = exp_output_dir + '/loss/loss_'  + exp_name + '_w.png'
    loss_weekly_df_name  = exp_output_dir + '/loss/loss_'  + exp_name + '_w.csv' 
    #==============================================================================================================#
    #================================================== csv file generator=========================================#
    #==============================================================================================================#
    train_csv, val_csv, test_csv = utils.data_generator(eval_scenario = scenario, spatial_resolution = spatial_resolution, 
                                                                patch_size = patch_size, patch_offset = patch_offset,  
                                                                cultivar_list = cultivar_list, 
                                                                year_list = year_list).return_split_dataframe()
    
    #==============================================================================================================#
    #============================================      Data Weight Generation     =================================#
    #==============================================================================================================#
    train_sampler, valid_sampler, test_sampler  = return_cost_sensitive_weight_sampler(train_csv, val_csv, test_csv, exp_output_dir, run_status = 'train')
    #==============================================================================================================#
    #============================================     Reading Data                =================================#
    #==============================================================================================================#

    if scenario == 'pixel_hold_out': 
        dataset_training = pixel_hold_out_dataloader(data_dir, exp_output_dir, category = 'train', patch_size = patch_size)
        dataset_validate = pixel_hold_out_dataloader(data_dir, exp_output_dir, category = 'val',  patch_size = patch_size)
        dataset_test     = pixel_hold_out_dataloader(data_dir, exp_output_dir, category = 'test',  patch_size = patch_size)
    else: 
        dataset_training = dataloader_(data_dir, exp_output_dir, category = 'train', patch_size = patch_size, in_channels = in_channel, 
                                                                                    spatial_resolution = spatial_resolution, run_status = 'valid')
        dataset_validate = dataloader_(data_dir, exp_output_dir, category = 'val',  patch_size = patch_size, in_channels = in_channel, 
                                                                                    spatial_resolution = spatial_resolution, run_status = 'valid')
        dataset_test     = dataloader_(data_dir, exp_output_dir, category = 'test',  patch_size = patch_size, in_channels = in_channel, 
                                                                                    spatial_resolution = spatial_resolution, run_status = 'eval')     
    #==============================================================================================================#
    #=============================================      Data Loader               =================================#
    #==============================================================================================================#                      
    # define training and validation data loaders
    data_loader_training = torch.utils.data.DataLoader(dataset_training, batch_size= batch_size, 
                                                    shuffle=False, sampler=train_sampler, num_workers=8) 
    data_loader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size= batch_size, 
                                                    shuffle=False, sampler=valid_sampler, num_workers=8)
    data_loader_test     = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                                    shuffle=False, sampler=test_sampler, num_workers=8) 
    #==============================================================================================================#
    #================================================ Model Calling ===============================================#
    #==============================================================================================================#

    model = UNet2DConvLSTM(in_channels = in_channel, out_channels = 1, 
                                                    num_filters   = 16, 
                                                    dropout       = dropout, 
                                                    Emb_Channels  = emb_channel, 
                                                    batch_size    = batch_size, 
                                                    botneck_size  = botneck_size).to(device)

    #================================================ Loss Function ===============================================#
    loss_fn = nn.MSELoss()
    #================================================ Optimizer ===================================================#
    ### ADAM
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay = weight_decay)
    #================================================= Training ===================================================#
    best_val_loss = 100000 # initial dummy value
    early_stopping = utils.EarlyStopping(tolerance=loss_stop_tolerance, min_delta=50)
    #==============================================================================================================#
    #==============================================================================================================#
    loss_stats = {'train': [],"val": []}

    loss_week_stats = {
        'train_w1': [], 'train_w2': [], 'train_w3': [], 'train_w4': [], 'train_w5': [],'train_w6': [],'train_w7': [],'train_w8': [], 'train_w9': [], 
        'train_w10': [], 'train_w11': [], 'train_w12': [], 'train_w13': [], 'train_w14': [], 'train_w15': [],
        "val_w1": [], "val_w2": [], "val_w3": [], "val_w4": [], "val_w5": [], "val_w6": [], "val_w7": [], "val_w8": [], "val_w9": [], "val_w10": [],
        "val_w11": [],"val_w12": [], "val_w13": [], "val_w14": [], "val_w15": []}
    for epoch in range(1, epochs+1):
        training_start_time = time.time()
        # TRAINING
        train_epoch_loss = 0
        tplw1, tplw2, tplw3, tplw4, tplw5, tplw6, tplw7, tplw8, tplw9, tplw10, tplw11, tplw12, tplw13, tplw14, tplw15 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        epoch_tr_weekly_loss_list = [tplw1, tplw2, tplw3, tplw4, tplw5, tplw6, tplw7, tplw8, tplw9, tplw10, tplw11, tplw12, tplw13, tplw14, tplw15]

        model.train()
        size = len(data_loader_training.dataset)

        for batch, sample in enumerate(data_loader_training):
            
            X_train_batch         = sample['image']
            y_train_batch         = sample['mask']
            y_train_batch         = y_train_batch[:,:,:,:,0]
            E_train_batch         = sample['EmbMatrix']

            X_train_batch, E_train_batch = X_train_batch.to(device), E_train_batch.to(device)
            #X_train_batch  = X_train_batch.to(device)
            y_train_batch  = y_train_batch.to(device)
                
            list_y_train_pred  = model(X_train_batch, E_train_batch)
            
            optimizer.zero_grad()
            
            train_loss_w1  = loss_fn(y_train_batch, list_y_train_pred[0])
            tplw1 += train_loss_w1.item()

            train_loss_w2  = loss_fn(y_train_batch, list_y_train_pred[1])
            tplw2 += train_loss_w2.item()
            
            train_loss_w3  = loss_fn(y_train_batch, list_y_train_pred[2])
            tplw3 += train_loss_w3.item()
            
            train_loss_w4  = loss_fn(y_train_batch, list_y_train_pred[3])
            tplw4 += train_loss_w4.item()
            
            train_loss_w5  = loss_fn(y_train_batch, list_y_train_pred[4])
            tplw5 += train_loss_w5.item()
            
            train_loss_w6  = loss_fn(y_train_batch, list_y_train_pred[5])
            tplw6 += train_loss_w6.item()
            
            train_loss_w7  = loss_fn(y_train_batch, list_y_train_pred[6])
            tplw7 += train_loss_w7.item()
            
            train_loss_w8  = loss_fn(y_train_batch, list_y_train_pred[7])
            tplw8 += train_loss_w8.item()
            
            train_loss_w9  = loss_fn(y_train_batch, list_y_train_pred[8])
            tplw9 += train_loss_w9.item()
            
            train_loss_w10 = loss_fn(y_train_batch, list_y_train_pred[9])
            tplw10 += train_loss_w10.item()
            
            train_loss_w11 = loss_fn(y_train_batch, list_y_train_pred[10])
            tplw11 += train_loss_w11.item()
            
            train_loss_w12 = loss_fn(y_train_batch, list_y_train_pred[11])
            tplw12 += train_loss_w12.item()
            
            train_loss_w13 = loss_fn(y_train_batch, list_y_train_pred[12])
            tplw13 += train_loss_w13.item()
            
            train_loss_w14 = loss_fn(y_train_batch, list_y_train_pred[13])
            tplw14 += train_loss_w14.item()
            
            train_loss_w15 = loss_fn(y_train_batch, list_y_train_pred[14])
            tplw15 += train_loss_w15.item()
            
            
            train_loss = train_loss_w1 + train_loss_w2 + train_loss_w3 + train_loss_w4 + train_loss_w5 + train_loss_w6 + train_loss_w7 + train_loss_w8 + train_loss_w9+ train_loss_w10+train_loss_w11+train_loss_w12+train_loss_w13+train_loss_w14+train_loss_w15
                
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            

        # VALIDATION    
        with torch.no_grad():
            
            val_epoch_loss = 0
            vplw1, vplw2, vplw3, vplw4, vplw5, vplw6, vplw7, vplw8, vplw9, vplw10, vplw11, vplw12, vplw13, vplw14, vplw15 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            model.eval()
            for batch, sample in enumerate(data_loader_validate):
                
                X_val_batch      = sample['image']
                y_val_batch      = sample['mask']
                y_val_batch      = y_val_batch[:,:,:,:,0]
                E_val_batch      = sample['EmbMatrix']
                
                X_val_batch, E_val_batch = X_val_batch.to(device), E_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)

                list_y_val_pred   = model(X_val_batch, E_val_batch)
                
                vl_w1 = loss_fn(y_val_batch, list_y_val_pred[0])
                vplw1 += vl_w1.item()

                vl_w2 = loss_fn(y_val_batch, list_y_val_pred[1])
                vplw2 += vl_w2.item()

                vl_w3 = loss_fn(y_val_batch, list_y_val_pred[2])
                vplw3 += vl_w3.item()

                vl_w4 = loss_fn(y_val_batch, list_y_val_pred[3])
                vplw4 += vl_w4.item()

                vl_w5 = loss_fn(y_val_batch, list_y_val_pred[4])
                vplw5 += vl_w5.item()

                vl_w6 = loss_fn(y_val_batch, list_y_val_pred[5])
                vplw6 += vl_w6.item()

                vl_w7 = loss_fn(y_val_batch, list_y_val_pred[6])
                vplw7 += vl_w7.item()

                vl_w8 = loss_fn(y_val_batch, list_y_val_pred[7])
                vplw8 += vl_w8.item() 

                vl_w9 = loss_fn(y_val_batch, list_y_val_pred[8])
                vplw9 += vl_w9.item()

                vl_w10 = loss_fn(y_val_batch, list_y_val_pred[9])
                vplw10 += vl_w10.item()

                vl_w11 = loss_fn(y_val_batch, list_y_val_pred[10])
                vplw11 += vl_w11.item()

                vl_w12 = loss_fn(y_val_batch, list_y_val_pred[11])
                vplw12 += vl_w12.item()

                vl_w13 = loss_fn(y_val_batch, list_y_val_pred[12])
                vplw13 += vl_w13.item()

                vl_w14 = loss_fn(y_val_batch, list_y_val_pred[13])
                vplw14 += vl_w14.item()

                vl_w15 = loss_fn(y_val_batch, list_y_val_pred[14])
                vplw15 += vl_w15.item()

                val_loss = vl_w1 + vl_w2 + vl_w3 + vl_w4 + vl_w5 + vl_w6 + vl_w7 + vl_w8 + vl_w9 + vl_w10 + vl_w11 + vl_w12 + vl_w13 + vl_w14 + vl_w15
                
                val_epoch_loss += val_loss.item()

        loss_stats['train'].append(train_epoch_loss/len(data_loader_training))
        loss_week_stats['train_w1'].append(tplw1/len(data_loader_training))
        loss_week_stats['train_w2'].append(tplw2/len(data_loader_training))
        loss_week_stats['train_w3'].append(tplw3/len(data_loader_training))
        loss_week_stats['train_w4'].append(tplw4/len(data_loader_training))
        loss_week_stats['train_w5'].append(tplw5/len(data_loader_training))
        loss_week_stats['train_w6'].append(tplw6/len(data_loader_training))
        loss_week_stats['train_w7'].append(tplw7/len(data_loader_training))
        loss_week_stats['train_w8'].append(tplw8/len(data_loader_training))
        loss_week_stats['train_w9'].append(tplw9/len(data_loader_training))
        loss_week_stats['train_w10'].append(tplw10/len(data_loader_training))
        loss_week_stats['train_w11'].append(tplw11/len(data_loader_training))
        loss_week_stats['train_w12'].append(tplw12/len(data_loader_training))
        loss_week_stats['train_w13'].append(tplw13/len(data_loader_training))
        loss_week_stats['train_w14'].append(tplw14/len(data_loader_training))
        loss_week_stats['train_w15'].append(tplw15/len(data_loader_training))
        
        loss_stats['val'].append(val_epoch_loss/len(data_loader_validate))
        loss_week_stats['val_w1'].append(vplw1/len(data_loader_validate))
        loss_week_stats['val_w2'].append(vplw2/len(data_loader_validate))
        loss_week_stats['val_w3'].append(vplw3/len(data_loader_validate))
        loss_week_stats['val_w4'].append(vplw4/len(data_loader_validate))
        loss_week_stats['val_w5'].append(vplw5/len(data_loader_validate))
        loss_week_stats['val_w6'].append(vplw6/len(data_loader_validate))
        loss_week_stats['val_w7'].append(vplw7/len(data_loader_validate))
        loss_week_stats['val_w8'].append(vplw8/len(data_loader_validate))
        loss_week_stats['val_w9'].append(vplw9/len(data_loader_validate))
        loss_week_stats['val_w10'].append(vplw10/len(data_loader_validate))
        loss_week_stats['val_w11'].append(vplw11/len(data_loader_validate))
        loss_week_stats['val_w12'].append(vplw12/len(data_loader_validate))
        loss_week_stats['val_w13'].append(vplw13/len(data_loader_validate))
        loss_week_stats['val_w14'].append(vplw14/len(data_loader_validate))
        loss_week_stats['val_w15'].append(vplw15/len(data_loader_validate))

        training_duration_time = (time.time() - training_start_time)        
        print(f'Epoch {epoch+0:03}: | Time(s): {training_duration_time:.3f}| Train Loss: {train_epoch_loss/len(data_loader_training):.4f} | Val Loss: {val_epoch_loss/len(data_loader_validate):.4f}') 
        
        if (val_epoch_loss/len(data_loader_validate)) < best_val_loss or epoch==0:
                    
            best_val_loss=(val_epoch_loss/len(data_loader_validate))
            torch.save(model.state_dict(), best_model_name)
            
            status = True
            
            bw1  = (vplw1/len(data_loader_validate))
            bw2  = (vplw2/len(data_loader_validate))
            bw3  = (vplw3/len(data_loader_validate))
            bw4  = (vplw4/len(data_loader_validate))
            bw5  = (vplw5/len(data_loader_validate))
            bw6  = (vplw6/len(data_loader_validate))
            bw7  = (vplw7/len(data_loader_validate))
            bw8  = (vplw8/len(data_loader_validate))
            bw9  = (vplw9/len(data_loader_validate))
            bw10 = (vplw10/len(data_loader_validate))
            bw11 = (vplw11/len(data_loader_validate))
            bw12 = (vplw12/len(data_loader_validate))
            bw13 = (vplw13/len(data_loader_validate))
            bw14 = (vplw14/len(data_loader_validate))
            bw15 = (vplw15/len(data_loader_validate))
            
            print(f'Best model Saved! Val MSE: {best_val_loss:.4f}')
            print(f'W1:{bw1:.2f}|W2:{bw2:.2f}|W3:{bw3:.2f}|W4:{bw4:.2f}|W5:{bw5:.2f}|W6:{bw6:.2f}|W7:{bw7:.2f}|W8:{bw8:.2f}|W9:{bw9:.2f}|W10:{bw10:.2f}|W11:{bw11:.2f}|W12:{bw12:.2f}|W13:{bw13:.2f}|W14:{bw14:.2f}|W15:{bw15:.2f}')
        else:
            print(f'Model is not saved! Current val Loss: {(val_epoch_loss/len(data_loader_validate)):.4f}') 
                
            status = False
            # early stopping
        early_stopping(status)
        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            break

    _ = ModelEngine.save_loss_df(loss_stats, loss_df_name, loss_fig_name)
    _ = ModelEngine.save_loss_df(loss_week_stats, loss_weekly_df_name, loss_weekly_fig_name)

    _ = ModelEngine.predict(model, data_loader_training, data_loader_validate, data_loader_test, exp_output_dir, Exp_name = exp_name)


if __name__ == "__main__":
    '''year_dict = {'Y1617':['2018', '2019', '2016', '2017'], 
                'Y1716':['2018', '2019', '2017', '2016'], 
            'Y1618':['2017', '2019', '2016', '2018'],
            'Y1816':['2017', '2019', '2018', '2016'],
            'Y1619':['2017', '2018', '2016', '2019'],
            'Y1916':['2017', '2018', '2019', '2016'],
            'Y1718':['2016', '2019', '2017', '2018'],
            'Y1817':['2016', '2019', '2018', '2017'],
            'Y1719':['2016', '2018', '2017', '2019'],
            'Y1917':['2016', '2018', '2019', '2017'],
            'Y1819':['2016', '2017', '2018', '2019'],
            'Y1918':['2016', '2017', '2019', '2018']}
    #for i in range(1, 5):
    for key, l in year_dict.items():'''
    wd_list = [0.9, 0.5, 0.1] 
    lr_list = [0.0001, 0.00001]
    for wd in wd_list: 
        for lr in lr_list: 
            ex_n = 'BYHO_6987' + '_' + str(wd)+ '_' + str(lr) 
            print(f"")
            train(scenario = 'block_year_hold_out', spatial_resolution = 10, patch_size = 16, patch_offset = 2,  
                cultivar_list = None,  year_list = ['2016', '2019', '2018', '2017'], 
                dropout = 0.40, batch_size = 64, learning_rate = lr, weight_decay = wd,
                in_channel = 5, emb_channel = 4, loss_stop_tolerance = 200, epochs = 400, exp_name = ex_n) 
