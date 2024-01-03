import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.set(font_scale=1.5)
sns.set_theme(style='white')
from mpl_toolkits.axes_grid1 import make_axes_locatable


from src.configs import blocks_information_dict, cultivars_, blocks_size



def ScenarioEvaluation2D(pred_npy):

    OutDF = pd.DataFrame()
    out_ytrue= []
    out_ypred_w1, out_ypred_w2, out_ypred_w3, out_ypred_w4, out_ypred_w5, out_ypred_w6, out_ypred_w7, out_ypred_w8 = [],[],[],[],[],[],[],[]
    out_ypred_w9, out_ypred_w10, out_ypred_w11, out_ypred_w12, out_ypred_w13, out_ypred_w14, out_ypred_w15 = [],[],[],[],[],[],[]
  
    for l in range(len(pred_npy)):

        block_key = next(iter(pred_npy[l]))
        block_allpatch = pred_npy[l][block_key]        
            
        ytrue     = pred_npy[l]['ytrue']
        ytrue_flat = ytrue.flatten()
        out_ytrue.append(ytrue_flat)
        
        ypred_w1  = pred_npy[l]['ypred_w1']
        ypred_w1_flat = ypred_w1.flatten()
        out_ypred_w1.append(ypred_w1_flat)
        
        ypred_w2  = pred_npy[l]['ypred_w2']
        ypred_w2_flat = ypred_w2.flatten()
        out_ypred_w2.append(ypred_w2_flat)
        
        ypred_w3  = pred_npy[l]['ypred_w3']
        ypred_w3_flat = ypred_w3.flatten()
        out_ypred_w3.append(ypred_w3_flat)
        
        ypred_w4  = pred_npy[l]['ypred_w4']
        ypred_w4_flat = ypred_w4.flatten()
        out_ypred_w4.append(ypred_w4_flat)
        
        ypred_w5  = pred_npy[l]['ypred_w5']
        ypred_w5_flat = ypred_w5.flatten()
        out_ypred_w5.append(ypred_w5_flat)
        
        ypred_w6  = pred_npy[l]['ypred_w6']
        ypred_w6_flat = ypred_w6.flatten()
        out_ypred_w6.append(ypred_w6_flat)
        
        ypred_w7  = pred_npy[l]['ypred_w7']
        ypred_w7_flat = ypred_w7.flatten()
        out_ypred_w7.append(ypred_w7_flat)
        
        ypred_w8  = pred_npy[l]['ypred_w8']
        ypred_w8_flat = ypred_w8.flatten()
        out_ypred_w8.append(ypred_w8_flat)
        
        ypred_w9  = pred_npy[l]['ypred_w9']
        ypred_w9_flat = ypred_w9.flatten()
        out_ypred_w9.append(ypred_w9_flat)
        
        ypred_w10  = pred_npy[l]['ypred_w10']
        ypred_w10_flat = ypred_w10.flatten()
        out_ypred_w10.append(ypred_w10_flat)
        
        ypred_w11  = pred_npy[l]['ypred_w11']
        ypred_w11_flat = ypred_w11.flatten()
        out_ypred_w11.append(ypred_w11_flat)
        
        ypred_w12  = pred_npy[l]['ypred_w12']
        ypred_w12_flat = ypred_w12.flatten()
        out_ypred_w12.append(ypred_w12_flat)
        
        ypred_w13  = pred_npy[l]['ypred_w13']
        ypred_w13_flat = ypred_w13.flatten()
        out_ypred_w13.append(ypred_w13_flat)
        
        ypred_w14  = pred_npy[l]['ypred_w14']
        ypred_w14_flat = ypred_w14.flatten()
        out_ypred_w14.append(ypred_w14_flat)
        
        ypred_w15  = pred_npy[l]['ypred_w15']
        ypred_w15_flat = ypred_w15.flatten()
        out_ypred_w15.append(ypred_w15_flat)
      


            
    out_ytrue = np.concatenate(out_ytrue)#.astype(np.float16)
    out_ypred_w1 = np.concatenate(out_ypred_w1) 
    out_ypred_w2 = np.concatenate(out_ypred_w2) 
    out_ypred_w3 = np.concatenate(out_ypred_w3) 
    out_ypred_w4 = np.concatenate(out_ypred_w4) 
    out_ypred_w5 = np.concatenate(out_ypred_w5) 
    out_ypred_w6 = np.concatenate(out_ypred_w6) 
    out_ypred_w7 = np.concatenate(out_ypred_w7) 
    out_ypred_w8 = np.concatenate(out_ypred_w8) 
    out_ypred_w9 = np.concatenate(out_ypred_w9) 
    out_ypred_w10 = np.concatenate(out_ypred_w10) 
    out_ypred_w11 = np.concatenate(out_ypred_w11) 
    out_ypred_w12 = np.concatenate(out_ypred_w12) 
    out_ypred_w13 = np.concatenate(out_ypred_w13) 
    out_ypred_w14 = np.concatenate(out_ypred_w14) 
    out_ypred_w15 = np.concatenate(out_ypred_w15) 

    
    OutDF['ytrue'] = out_ytrue
    OutDF['ypred_w1'] = out_ypred_w1
    OutDF['ypred_w2'] = out_ypred_w2 
    OutDF['ypred_w3'] = out_ypred_w3 
    OutDF['ypred_w4'] = out_ypred_w4 
    OutDF['ypred_w5'] = out_ypred_w5 
    OutDF['ypred_w6'] = out_ypred_w6 
    OutDF['ypred_w7'] = out_ypred_w7 
    OutDF['ypred_w8'] = out_ypred_w8 
    OutDF['ypred_w9'] = out_ypred_w9 
    OutDF['ypred_w10'] = out_ypred_w10 
    OutDF['ypred_w11'] = out_ypred_w11 
    OutDF['ypred_w12'] = out_ypred_w12 
    OutDF['ypred_w13'] = out_ypred_w13 
    OutDF['ypred_w14'] = out_ypred_w14 
    OutDF['ypred_w15'] = out_ypred_w15


    return OutDF

def npy_block_names(npy_array):

    block_names = []
    for i in range(len(npy_array)):
        blocks = npy_array[i]['block']
        block_names.append(blocks)

    np_arr = np.concatenate(block_names)
    out = np.unique(np_arr)
    
    return out

def xy_vector_generator(x0, y0, wsize):
    x_vector, y_vector = [], []
    
    for i in range(x0, x0+wsize):
        for j in range(y0, y0+wsize):
            x_vector.append(i)
            y_vector.append(j)

    return x_vector, y_vector 

def time_series_eval_csv(pred_npy, blocks_list, wsize = None):
    #blocks_list = get_blocks_from_patches(pred_npy)
    
    OutDF = pd.DataFrame()
    out_ytrue, out_blocks, out_cultivars, out_x, out_y = [], [], [], [], []
    out_ypred_w1, out_ypred_w2,out_ypred_w3,out_ypred_w4,out_ypred_w5 = [], [], [], [], []
    out_ypred_w6, out_ypred_w7,out_ypred_w8,out_ypred_w9,out_ypred_w10 = [], [], [], [], []
    out_ypred_w11, out_ypred_w12,out_ypred_w13,out_ypred_w14,out_ypred_w15 = [], [], [], [], []
    
    for block in blocks_list:  
        
        name_split = os.path.split(block)[-1]
        block_name = name_split.replace(name_split[7:], '')
        root_name  = name_split.replace(name_split[:4], '').replace(name_split[3], '')
        block_id   = root_name
        
        res           = {key: blocks_information_dict[key] for key in blocks_information_dict.keys() & {block_name}}
        list_d        = res.get(block_name)
        cultivar_id   = list_d[1]

        
        for l in range(len(pred_npy)):
            tb_pred_indices = [i for i, x in enumerate(pred_npy[l]['block']) if x == block]
            if len(tb_pred_indices) !=0:   
                for index in tb_pred_indices:

                    x0                = pred_npy[l]['X'][index]
                    y0                = pred_npy[l]['Y'][index]
                    x_vector, y_vector = xy_vector_generator(x0, y0, wsize)
                    out_x.append(x_vector)
                    out_y.append(y_vector)
       
                    tb_ytrue         = pred_npy[l]['ytrue'][index]
                    tb_flatten_ytrue = tb_ytrue.flatten()
                    out_ytrue.append(tb_flatten_ytrue)
                    

                    tb_ypred_w1    = pred_npy[l]['ypred_w1'][index]
                    tb_flatten_ypred_w1 = tb_ypred_w1.flatten()
                    out_ypred_w1.append(tb_flatten_ypred_w1)
                    
                    tb_ypred_w2    = pred_npy[l]['ypred_w2'][index]
                    tb_flatten_ypred_w2 = tb_ypred_w2.flatten()
                    out_ypred_w2.append(tb_flatten_ypred_w2)
                    
                    tb_ypred_w3    = pred_npy[l]['ypred_w3'][index]
                    tb_flatten_ypred_w3 = tb_ypred_w3.flatten()
                    out_ypred_w3.append(tb_flatten_ypred_w3)
                    
                    tb_ypred_w4    = pred_npy[l]['ypred_w4'][index]
                    tb_flatten_ypred_w4 = tb_ypred_w4.flatten()
                    out_ypred_w4.append(tb_flatten_ypred_w4)
                    
                    tb_ypred_w5    = pred_npy[l]['ypred_w5'][index]
                    tb_flatten_ypred_w5 = tb_ypred_w5.flatten()
                    out_ypred_w5.append(tb_flatten_ypred_w5)

                    tb_ypred_w6    = pred_npy[l]['ypred_w6'][index]
                    tb_flatten_ypred_w6 = tb_ypred_w6.flatten()
                    out_ypred_w6.append(tb_flatten_ypred_w6)
                    
                    tb_ypred_w7    = pred_npy[l]['ypred_w7'][index]
                    tb_flatten_ypred_w7 = tb_ypred_w7.flatten()
                    out_ypred_w7.append(tb_flatten_ypred_w7)
                    
                    tb_ypred_w8    = pred_npy[l]['ypred_w8'][index]
                    tb_flatten_ypred_w8 = tb_ypred_w8.flatten()
                    out_ypred_w8.append(tb_flatten_ypred_w8)
                    
                    tb_ypred_w9    = pred_npy[l]['ypred_w9'][index]
                    tb_flatten_ypred_w9 = tb_ypred_w9.flatten()
                    out_ypred_w9.append(tb_flatten_ypred_w9)
                    
                    tb_ypred_w10    = pred_npy[l]['ypred_w10'][index]
                    tb_flatten_ypred_w10 = tb_ypred_w10.flatten()
                    out_ypred_w10.append(tb_flatten_ypred_w10)
                    
                    tb_ypred_w11    = pred_npy[l]['ypred_w11'][index]
                    tb_flatten_ypred_w11 = tb_ypred_w11.flatten()
                    out_ypred_w11.append(tb_flatten_ypred_w11)
                    
                    tb_ypred_w12    = pred_npy[l]['ypred_w12'][index]
                    tb_flatten_ypred_w12 = tb_ypred_w12.flatten()
                    out_ypred_w12.append(tb_flatten_ypred_w12)
                    
                    tb_ypred_w13    = pred_npy[l]['ypred_w13'][index]
                    tb_flatten_ypred_w13 = tb_ypred_w13.flatten()
                    out_ypred_w13.append(tb_flatten_ypred_w13)
                    
                    tb_ypred_w14    = pred_npy[l]['ypred_w14'][index]
                    tb_flatten_ypred_w14 = tb_ypred_w14.flatten()
                    out_ypred_w14.append(tb_flatten_ypred_w14)
                    
                    tb_ypred_w15    = pred_npy[l]['ypred_w15'][index]
                    tb_flatten_ypred_w15 = tb_ypred_w15.flatten()
                    out_ypred_w15.append(tb_flatten_ypred_w15)
                    
                    tb_block_id   = np.array(len(tb_flatten_ytrue)*[block_id], dtype=np.int32)
                    out_blocks.append(tb_block_id)

                    tb_cultivar_id = np.array(len(tb_flatten_ytrue)*[cultivar_id], dtype=np.int8)
                    out_cultivars.append(tb_cultivar_id)


                    
    # agg    
    out_blocks        = np.concatenate(out_blocks)
    out_cultivars     = np.concatenate(out_cultivars)
    out_x             = np.concatenate(out_x)
    out_y             = np.concatenate(out_y)
    out_ytrue         = np.concatenate(out_ytrue)
    out_ypred_w1         = np.concatenate(out_ypred_w1)
    out_ypred_w2         = np.concatenate(out_ypred_w2)
    out_ypred_w3         = np.concatenate(out_ypred_w3)
    out_ypred_w4         = np.concatenate(out_ypred_w4)
    out_ypred_w5         = np.concatenate(out_ypred_w5)
    out_ypred_w6         = np.concatenate(out_ypred_w6)
    out_ypred_w7         = np.concatenate(out_ypred_w7)
    out_ypred_w8         = np.concatenate(out_ypred_w8)
    out_ypred_w9         = np.concatenate(out_ypred_w9)
    out_ypred_w10         = np.concatenate(out_ypred_w10)
    out_ypred_w11         = np.concatenate(out_ypred_w11)
    out_ypred_w12         = np.concatenate(out_ypred_w12)
    out_ypred_w13         = np.concatenate(out_ypred_w13)
    out_ypred_w14         = np.concatenate(out_ypred_w14)
    out_ypred_w15         = np.concatenate(out_ypred_w15)
    
    OutDF['block']    = out_blocks
    OutDF['cultivar'] = out_cultivars
    OutDF['x']        = out_x
    OutDF['y']        = out_y
    OutDF['ytrue']    = out_ytrue
    OutDF['ypred_w1']    = out_ypred_w1
    OutDF['ypred_w2']    = out_ypred_w2
    OutDF['ypred_w3']    = out_ypred_w3
    OutDF['ypred_w4']    = out_ypred_w4
    OutDF['ypred_w5']    = out_ypred_w5
    OutDF['ypred_w6']    = out_ypred_w6
    OutDF['ypred_w7']    = out_ypred_w7
    OutDF['ypred_w8']    = out_ypred_w8
    OutDF['ypred_w9']    = out_ypred_w9
    OutDF['ypred_w10']    = out_ypred_w10
    OutDF['ypred_w11']    = out_ypred_w11
    OutDF['ypred_w12']    = out_ypred_w12
    OutDF['ypred_w13']    = out_ypred_w13
    OutDF['ypred_w14']    = out_ypred_w14
    OutDF['ypred_w15']    = out_ypred_w15
    
    NewOUtDF = agg_pixelovelapp_df_2d(OutDF)
    
    return OutDF, NewOUtDF

def agg_pixelovelapp_df_2d(df):
    out = pd.DataFrame()
    
    newdf = df.groupby(["block", "cultivar", "x", "y"]).agg(
        block    = pd.NamedAgg(column="block", aggfunc=np.unique),
        cultivar = pd.NamedAgg(column="cultivar", aggfunc=np.unique),
        x        = pd.NamedAgg(column="x", aggfunc=np.unique),
        y        = pd.NamedAgg(column="y", aggfunc=np.unique),
        ytrue    = pd.NamedAgg(column="ytrue", aggfunc=np.mean),
        ypred_w1    = pd.NamedAgg(column="ypred_w1", aggfunc=np.mean),
        ypred_w2    = pd.NamedAgg(column="ypred_w2", aggfunc=np.mean),
        ypred_w3    = pd.NamedAgg(column="ypred_w3", aggfunc=np.mean),
        ypred_w4    = pd.NamedAgg(column="ypred_w4", aggfunc=np.mean),
        ypred_w5    = pd.NamedAgg(column="ypred_w5", aggfunc=np.mean),
        ypred_w6    = pd.NamedAgg(column="ypred_w6", aggfunc=np.mean),
        ypred_w7    = pd.NamedAgg(column="ypred_w7", aggfunc=np.mean),
        ypred_w8    = pd.NamedAgg(column="ypred_w8", aggfunc=np.mean),
        ypred_w9    = pd.NamedAgg(column="ypred_w9", aggfunc=np.mean),
        ypred_w10    = pd.NamedAgg(column="ypred_w10", aggfunc=np.mean),
        ypred_w11    = pd.NamedAgg(column="ypred_w11", aggfunc=np.mean),
        ypred_w12    = pd.NamedAgg(column="ypred_w12", aggfunc=np.mean),
        ypred_w13    = pd.NamedAgg(column="ypred_w13", aggfunc=np.mean),
        ypred_w14    = pd.NamedAgg(column="ypred_w14", aggfunc=np.mean),
        ypred_w15    = pd.NamedAgg(column="ypred_w15", aggfunc=np.mean),
        
    )
    
    
    out['block'] = newdf['block'].values
    out['cultivar'] = newdf['cultivar'].values
    out['x'] = newdf['x'].values
    out['y'] = newdf['y'].values
    out['ytrue'] = newdf['ytrue'].values
    out['ypred_w1'] = newdf['ypred_w1'].values
    out['ypred_w2'] = newdf['ypred_w2'].values
    out['ypred_w3'] = newdf['ypred_w3'].values
    out['ypred_w4'] = newdf['ypred_w4'].values
    out['ypred_w5'] = newdf['ypred_w5'].values
    out['ypred_w6'] = newdf['ypred_w6'].values
    out['ypred_w7'] = newdf['ypred_w7'].values
    out['ypred_w8'] = newdf['ypred_w8'].values
    out['ypred_w9'] = newdf['ypred_w9'].values
    out['ypred_w10'] = newdf['ypred_w10'].values
    out['ypred_w11'] = newdf['ypred_w11'].values
    out['ypred_w12'] = newdf['ypred_w12'].values
    out['ypred_w13'] = newdf['ypred_w13'].values
    out['ypred_w14'] = newdf['ypred_w14'].values
    out['ypred_w15'] = newdf['ypred_w15'].values


    return out

def time_series_evaluation_plots(train, val, test, fig_save_name): 
    
    Weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']
    
    results = pd.DataFrame()
    train_r2_list, train_mae_list, train_rmse_list, train_mape_list = [], [], [], [] 
    val_r2_list, val_mae_list, val_rmse_list, val_mape_list         = [], [], [], []
    test_r2_list, test_mae_list, test_rmse_list, test_mape_list     = [], [], [], []
    
    for i in range(len(Weeks)): 
        train_r2, train_mae, train_rmse, train_mape, _, _ = regression_metrics(train['ytrue'], train.iloc[:, i+1])
        train_r2_list.append(train_r2)
        train_mae_list.append(train_mae)
        train_rmse_list.append(train_rmse)
        train_mape_list.append(train_mape)
        
        val_r2, val_mae, val_rmse, val_mape, _, _         = regression_metrics(val['ytrue'], val.iloc[:, i+1])
        val_r2_list.append(val_r2)
        val_mae_list.append(val_mae)
        val_rmse_list.append(val_rmse)
        val_mape_list.append(val_mape)
        
        test_r2, test_mae, test_rmse, test_mape, _, _     = regression_metrics(test['ytrue'], test.iloc[:, i+1])
        test_r2_list.append(test_r2)
        test_mae_list.append(test_mae)
        test_rmse_list.append(test_rmse)
        test_mape_list.append(test_mape)
    
    results['weeks']      = Weeks
    results['Train_R2']   = train_r2_list
    results['Valid_R2']   = val_r2_list
    results['Test_R2']    = test_r2_list
    results['Train_MAE']  = train_mae_list
    results['Valid_MAE']  = val_mae_list
    results['Test_MAE']   = test_mae_list
    results['Train_RMSE'] = train_rmse_list
    results['Valid_RMSE'] = val_rmse_list
    results['Test_RMSE']  = test_rmse_list
    results['Train_MAPE'] = train_mape_list
    results['Valid_MAPE'] = val_mape_list
    results['Test_MAPE']  = test_mape_list
 

    #plt.rcParams["axes.grid"] = True
    fig, axs = plt.subplots(2, 2, figsize = (20, 10), sharex=True)


    axs[0, 0].plot(results["weeks"], results["Train_R2"], "-o")
    axs[0, 0].plot(results["weeks"], results["Valid_R2"], "-*")
    axs[0, 0].plot(results["weeks"], results["Test_R2"],  "-d")
    axs[0, 0].set_ylabel('R2')
    axs[0, 0].set_facecolor('white')
    plt.setp(axs[0, 0].spines.values(), color='k')
    
    axs[0, 1].plot(results["weeks"], results["Train_RMSE"], "-o")
    axs[0, 1].plot(results["weeks"], results["Valid_RMSE"], "-*")
    axs[0, 1].plot(results["weeks"], results["Test_RMSE"],  "-d")
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].set_facecolor('white')
    plt.setp(axs[0, 1].spines.values(), color='k')
    
    axs[1, 0].plot(results["weeks"], results["Train_MAE"], "-o")
    axs[1, 0].plot(results["weeks"], results["Valid_MAE"], "-*")
    axs[1, 0].plot(results["weeks"], results["Test_MAE"],  "-d")
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].set_ylabel('MAE (t/ha)')
    axs[1, 0].set_facecolor('white')
    plt.setp(axs[1, 0].spines.values(), color='k')
    
    axs[1, 1].plot(results["weeks"], results["Train_MAPE"], "-o", label = 'train')
    axs[1, 1].plot(results["weeks"], results["Valid_MAPE"], "-*", label = 'valid')
    axs[1, 1].plot(results["weeks"], results["Test_MAPE"],  "-d", label = 'test')
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].set_ylabel('MAPE (%)')
    axs[1, 1].set_facecolor('white')
    plt.setp(axs[1, 1].spines.values(), color='k')
    axs[1, 1].legend(loc="upper right")
    
    
    

    plt.savefig(fig_save_name, dpi = 300)
    
    return results 

def time_series_year_hold_out_evaluation(df2016, df2017, df2018, df2019): 
    Weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']

    #plt.rcParams["axes.grid"] = True
    fig, axs = plt.subplots(4, 2, figsize = (20, 20), sharex=True)


    axs[0, 0].plot(df2016["weeks"], df2016["Train_MAE_M"], "-o")
    axs[0, 0].fill_between(df2016["weeks"], df2016["Train_MAE_M"] - df2016['Train_MAE_S'], df2016["Train_MAE_M"]+ df2016['Train_MAE_S'], alpha=.2)
    axs[0, 0].plot(df2016["weeks"], df2016["Valid_MAE_M"], "-*")
    axs[0, 0].fill_between(df2016["weeks"], df2016["Valid_MAE_M"] - df2016['Valid_MAE_S'], df2016["Valid_MAE_M"]+ df2016['Valid_MAE_S'], alpha=.2)
    axs[0, 0].plot(df2016["weeks"], df2016["Test_MAE_M"],  "-d")
    axs[0, 0].fill_between(df2016["weeks"], df2016["Test_MAE_M"] - df2016['Test_MAE_S'], df2016["Test_MAE_M"]+ df2016['Test_MAE_S'], alpha=.2)
    axs[0, 0].set_ylabel('MAE (ton/ac), 2016')
    axs[0, 0].set_facecolor('white')
    axs[0, 0].set_ylim([1, 3])
    plt.setp(axs[0, 0].spines.values(), color='k')
    
    axs[0, 1].plot(df2016["weeks"], df2016["Train_MAPE_M"], "-o", label = 'train')
    axs[0, 1].fill_between(df2016["weeks"], df2016["Train_MAPE_M"] - df2016['Train_MAPE_S'], df2016["Train_MAPE_M"]+ df2016['Train_MAPE_S'], alpha=.2)
    axs[0, 1].plot(df2016["weeks"], df2016["Valid_MAPE_M"], "-*", label = 'valid')
    axs[0, 1].fill_between(df2016["weeks"], df2016["Valid_MAPE_M"] - df2016['Valid_MAPE_S'], df2016["Valid_MAPE_M"]+ df2016['Valid_MAPE_S'], alpha=.2)
    axs[0, 1].plot(df2016["weeks"], df2016["Test_MAPE_M"],  "-d", label = 'test')
    axs[0, 1].fill_between(df2016["weeks"], df2016["Test_MAPE_M"] - df2016['Test_MAPE_S'], df2016["Test_MAPE_M"]+ df2016['Test_MAPE_S'], alpha=.2)
    axs[0, 1].set_ylabel('MAPE (%)')
    axs[0, 1].set_facecolor('white')
    axs[0, 1].set_ylim([0.06, 0.28])
    plt.setp(axs[0, 1].spines.values(), color='k')
    axs[0, 1].legend(loc="upper right")
    
    axs[1, 0].plot(df2017["weeks"], df2017["Train_MAE_M"], "-o")
    axs[1, 0].fill_between(df2017["weeks"], df2017["Train_MAE_M"] - df2017['Train_MAE_S'], df2017["Train_MAE_M"]+ df2017['Train_MAE_S'], alpha=.2)
    axs[1, 0].plot(df2017["weeks"], df2017["Valid_MAE_M"], "-*")
    axs[1, 0].fill_between(df2017["weeks"], df2017["Valid_MAE_M"] - df2017['Valid_MAE_S'], df2017["Valid_MAE_M"]+ df2017['Valid_MAE_S'], alpha=.2)
    axs[1, 0].plot(df2017["weeks"], df2017["Test_MAE_M"],  "-d")
    axs[1, 0].fill_between(df2017["weeks"], df2017["Test_MAE_M"] - df2017['Test_MAE_S'], df2017["Test_MAE_M"]+ df2017['Test_MAE_S'], alpha=.2)
    axs[1, 0].set_ylabel('MAE (ton/ac), 2017')
    axs[1, 0].set_facecolor('white')
    axs[1, 0].set_ylim([1, 3])
    plt.setp(axs[1, 0].spines.values(), color='k')
    
    axs[1, 1].plot(df2017["weeks"], df2017["Train_MAPE_M"], "-o")
    axs[1, 1].fill_between(df2017["weeks"], df2017["Train_MAPE_M"] - df2017['Train_MAPE_S'], df2017["Train_MAPE_M"]+ df2017['Train_MAPE_S'], alpha=.2)
    axs[1, 1].plot(df2017["weeks"], df2017["Valid_MAPE_M"], "-*")
    axs[1, 1].fill_between(df2017["weeks"], df2017["Valid_MAPE_M"] - df2017['Valid_MAPE_S'], df2017["Valid_MAPE_M"]+ df2017['Valid_MAPE_S'], alpha=.2)
    axs[1, 1].plot(df2017["weeks"], df2017["Test_MAPE_M"],  "-d")
    axs[1, 1].fill_between(df2017["weeks"], df2017["Test_MAPE_M"] - df2017['Test_MAPE_S'], df2017["Test_MAPE_M"]+ df2017['Test_MAPE_S'], alpha=.2)
    axs[1, 1].set_ylabel('MAPE (%)')
    axs[1, 1].set_facecolor('white')
    axs[1, 1].set_ylim([0.06, 0.28])
    plt.setp(axs[1, 1].spines.values(), color='k')

    axs[2, 0].plot(df2018["weeks"], df2018["Train_MAE_M"], "-o")
    axs[2, 0].fill_between(df2018["weeks"], df2018["Train_MAE_M"] - df2018['Train_MAE_S'], df2018["Train_MAE_M"]+ df2018['Train_MAE_S'], alpha=.2)
    axs[2, 0].plot(df2018["weeks"], df2018["Valid_MAE_M"], "-*")
    axs[2, 0].fill_between(df2018["weeks"], df2018["Valid_MAE_M"] - df2018['Valid_MAE_S'], df2018["Valid_MAE_M"]+ df2018['Valid_MAE_S'], alpha=.2)
    axs[2, 0].plot(df2018["weeks"], df2018["Test_MAE_M"],  "-d")
    axs[2, 0].fill_between(df2018["weeks"], df2018["Test_MAE_M"] - df2018['Test_MAE_S'], df2018["Test_MAE_M"]+ df2018['Test_MAE_S'], alpha=.2)
    axs[2, 0].set_ylabel('MAE (ton/ac), 2018')
    axs[2, 0].set_facecolor('white')
    axs[2, 0].set_ylim([1, 3])
    plt.setp(axs[2, 0].spines.values(), color='k')
    
    axs[2, 1].plot(df2018["weeks"], df2018["Train_MAPE_M"], "-o")
    axs[2, 1].fill_between(df2018["weeks"], df2018["Train_MAPE_M"] - df2018['Train_MAPE_S'], df2018["Train_MAPE_M"]+ df2018['Train_MAPE_S'], alpha=.2)
    axs[2, 1].plot(df2018["weeks"], df2018["Valid_MAPE_M"], "-*")
    axs[2, 1].fill_between(df2018["weeks"], df2018["Valid_MAPE_M"] - df2018['Valid_MAPE_S'], df2018["Valid_MAPE_M"]+ df2018['Valid_MAPE_S'], alpha=.2)
    axs[2, 1].plot(df2018["weeks"], df2018["Test_MAPE_M"],  "-d")
    axs[2, 1].fill_between(df2018["weeks"], df2018["Test_MAPE_M"]  - df2018['Test_MAPE_S'], df2018["Test_MAPE_M"] + df2018['Test_MAPE_S'], alpha=.2)
    axs[2, 1].set_ylabel('MAPE (%)')
    axs[2, 1].set_facecolor('white')
    axs[2, 1].set_ylim([0.06, 0.28])
    plt.setp(axs[2, 1].spines.values(), color='k')

    axs[3, 0].plot(df2019["weeks"], df2019["Train_MAE_M"], "-o")
    axs[3, 0].fill_between(df2019["weeks"], df2019["Train_MAE_M"] - df2019['Train_MAE_S'], df2019["Train_MAE_M"]+ df2019['Train_MAE_S'], alpha=.2)
    axs[3, 0].plot(df2019["weeks"], df2019["Valid_MAE_M"], "-*")
    axs[3, 0].fill_between(df2019["weeks"], df2019["Valid_MAE_M"] - df2019['Train_MAE_S'], df2019["Valid_MAE_M"]+ df2019['Train_MAE_S'], alpha=.2)
    axs[3, 0].plot(df2019["weeks"], df2019["Test_MAE_M"],  "-d")
    axs[3, 0].fill_between(df2019["weeks"], df2019["Test_MAE_M"] - df2019['Test_MAE_S'], df2019["Test_MAE_M"]+ df2019['Test_MAE_S'], alpha=.2)
    axs[3, 0].set_ylabel('MAE (ton/ac), 2019')
    axs[3, 0].set_facecolor('white')
    axs[3, 0].tick_params(axis='x', rotation=45)
    axs[3, 0].set_ylim([1, 3])
    plt.setp(axs[3, 0].spines.values(), color='k')
    
    axs[3, 1].plot(df2019["weeks"], df2019["Train_MAPE_M"], "-o")
    axs[3, 1].fill_between(df2019["weeks"], df2019["Train_MAPE_M"] - df2019['Train_MAPE_S'], df2019["Train_MAPE_M"]+ df2019['Train_MAPE_S'], alpha=.2)
    axs[3, 1].plot(df2019["weeks"], df2019["Valid_MAPE_M"], "-*")
    axs[3, 1].fill_between(df2019["weeks"], df2019["Valid_MAPE_M"] - df2019['Valid_MAPE_S'], df2019["Valid_MAPE_M"]+ df2019['Valid_MAPE_S'], alpha=.2)
    axs[3, 1].plot(df2019["weeks"], df2019["Test_MAPE_M"],  "-d")
    axs[3, 1].fill_between(df2019["weeks"], df2019["Test_MAPE_M"] - df2019['Test_MAPE_S'], df2019["Test_MAPE_M"]+ df2019['Test_MAPE_S'], alpha=.2)
    axs[3, 1].set_ylabel('MAPE (%)')
    axs[3, 1].set_facecolor('white')
    axs[3, 1].tick_params(axis='x', rotation=45)
    axs[3, 1].set_ylim([0.06, 0.28])
    plt.setp(axs[3, 1].spines.values(), color='k')

def regression_metrics(ytrue, ypred):
    """Calculating the evaluation metric based on regression results:
    input:
            ytrue: 
            ypred:
            
    output:
            root mean square error(rmse)
            root square(r^2)
            mean absolute error(mae)
            mean absolute percent error(mape)
            mean of true yield
            mean of prediction yield
    """
    mean_ytrue = np.mean(ytrue)
    mean_ypred = np.mean(ypred)
    rmse       = np.sqrt(mean_squared_error(ytrue, ypred))
    mape       = mean_absolute_percentage_error(ytrue, ypred)
    r_square   = r2_score(ytrue, ypred)
    mae        = mean_absolute_error(ytrue, ypred)
    
    return [r_square , mae, rmse, mape, mean_ytrue, mean_ypred] 

def block_cultivar_test_csv_results_2d(test_df, week = None, save_dir= None, save_csv_name = None):

    test_blocks     = test_df.groupby(by = ['block'])

    block_df    = pd.DataFrame()
    b, c, test_r2_b, test_mae_b, test_rmse_b, test_mape_b = [], [], [], [], [], [] 
    
    cultivar_df = pd.DataFrame()
    cu, test_r2_c, test_mae_c, test_rmse_c, test_mape_c   = [], [], [], [], []
        
    for block, tedf in test_blocks:
        b.append(block)
        cultivar_id     = tedf.iloc[0]['cultivar']
        cultivar_id     = int(cultivar_id)
        cultivar_id     = str(cultivar_id)
        key_ext         = {key: cultivars_[key] for key in cultivars_.keys() & {cultivar_id}}
        #print(key_ext)
        list_d          = key_ext.get(cultivar_id)
        cultivar_type   = list_d[0]
        
        c.append(cultivar_type)
        b_r_square3, b_mae3, b_rmse3, b_mape3, mean_ytrue, mean_ypred =  regression_metrics(tedf['ytrue'], tedf[week])
        test_r2_b.append(b_r_square3)
        test_mae_b.append(b_mae3)
        test_rmse_b.append(b_rmse3)
        test_mape_b.append(b_mape3)
        
        
    block_df['block']    = b
    block_df['cultivar'] = c
    block_df['R2']  = test_r2_b
    block_df['MAE']  = test_mae_b
    block_df['RMSE'] = test_rmse_b
    block_df['MAPE'] = test_mape_b
    block_df = block_df.round(decimals = 2)
    block_df.to_csv(os.path.join(save_dir, save_csv_name + '_blocks.csv'))
    
    ### Cultvars 
    test_cultivars  = test_df.groupby(by = ['cultivar'])
    for cul, tedf1 in test_cultivars:
        cultivar_id   = int(cul)
        cultivar_id   = str(cultivar_id)
        key_ext       = {key: cultivars_[key] for key in cultivars_.keys() & {cultivar_id}}
        #print(key_ext)
        list_d        = key_ext.get(cultivar_id)
        cultivar_type = list_d[0]
        
        cu.append(cultivar_type)
        b_r_square33, b_mae33, b_rmse33, b_mape33, mean_ytrue, mean_ypred =  regression_metrics(tedf1['ytrue'], tedf1[week])
        test_r2_c.append(b_r_square33)
        test_mae_c.append(b_mae33)
        test_rmse_c.append(b_rmse33)
        test_mape_c.append(b_mape33)

    cultivar_df['cultivar'] = cu
    cultivar_df['R2']  = test_r2_c
    cultivar_df['MAE']  = test_mae_c
    cultivar_df['RMSE'] = test_rmse_c
    cultivar_df['MAPE'] = test_mape_c
    cultivar_df = cultivar_df.round(decimals = 2)
    cultivar_df.to_csv(os.path.join(save_dir, save_csv_name + '_cultivars.csv'))
    
    return block_df, cultivar_df

def train_val_test_satterplot(train_df, valid_df, test_df, week = None, cmap  = None, mincnt = None):

    
    if week == None: 
        week_pred = 'ypred'
    else: 
        week_pred = 'ypred_w' + str(week)

    w_train_e1  = train_df[['ytrue', week_pred]]
    w_train_e1  = w_train_e1.rename(columns={week_pred: "ypred"})
    tarin_true  = w_train_e1['ytrue']
    train_pred  = w_train_e1['ypred']
    tr_r2, tr_mae, tr_rmse, tr_mape, _,_ = regression_metrics(tarin_true, train_pred)

    TR = sns.jointplot(x=tarin_true, y=train_pred, kind="hex", height=8, ratio=4,  
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = cmap , mincnt=mincnt, joint_kws={"facecolor": 'white'})#,,  marginal_kws = dict(bins = np.arange(0, 50000))

    for patch in TR.ax_marg_x.patches:
        patch.set_facecolor('grey')

    for patch in TR.ax_marg_y.patches:
        patch.set_facecolor('grey')

    TR.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)

    plt.xlabel('Measured (ton/ac) - Train')
    plt.ylabel('Predict (ton/ac)')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(tr_r2, tr_mae, tr_rmse, tr_mape)
    plt.legend([extra], [scores], loc='upper left')
    #plt.title('Train Data')
    #========================================================
    w_valid_e1  = valid_df[['ytrue', week_pred]]
    w_valid_e1  = w_valid_e1.rename(columns={week_pred: "ypred"})
    valid_true = w_valid_e1['ytrue']
    valid_pred = w_valid_e1['ypred']
    val_r2, val_mae, val_rmse, val_mape, _,_ = regression_metrics(valid_true, valid_pred)

    Va = sns.jointplot(x = valid_true, y = valid_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = 'viridis', mincnt = mincnt) #palette ='flare', ,   
    for patch in Va.ax_marg_x.patches:
        patch.set_facecolor('grey')
    for patch in Va.ax_marg_y.patches:
        patch.set_facecolor('grey')


    Va.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)
    plt.xlabel('Measured (ton/ac) - Validation')
    plt.ylabel('')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(val_r2, val_mae, val_rmse, val_mape)
    plt.legend([extra], [scores], loc='upper left')
    #plt.title('Validation Data')
    #========================================================
    w_test_e1  = test_df[['ytrue', week_pred]]
    w_test_e1  = w_test_e1.rename(columns={week_pred: "ypred"})
    test_true = w_test_e1['ytrue']
    valid_pred = w_test_e1['ypred']
    test_r2, test_mae, test_rmse, test_mape, _,_ = regression_metrics(test_true, valid_pred)

    Te = sns.jointplot(x=test_true, y = valid_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = cmap, mincnt = mincnt) #palette ='flare', ,  
    for patch in Te.ax_marg_x.patches:
        patch.set_facecolor('grey')
    for patch in Te.ax_marg_y.patches:
        patch.set_facecolor('grey') 

    Te.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)

    plt.xlabel('Measured (ton/ac)')
    plt.ylabel('')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(test_r2, test_mae, test_rmse, test_mape)
    plt.legend([extra], [scores], loc='upper left')



    fig = plt.figure(figsize=(21,7))
    gs = gridspec.GridSpec(1, 3)

    mg0 = SeabornFig2Grid(TR, fig, gs[0])
    mg1 = SeabornFig2Grid(Va, fig, gs[1])
    mg2 = SeabornFig2Grid(Te, fig, gs[2])


    gs.tight_layout(fig)
    #gs.update(top=0.7)

    plt.show()

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

def thriple_time_series_block_eval(df1, df2, df3): 
    Weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']
    
    results = pd.DataFrame()
    test_mae_list1, test_mape_list1     = [], []
    test_mae_list2, test_mape_list2     = [], []
    test_mae_list3, test_mape_list3     = [], []
    
    for i in range(len(Weeks)): 
        test_r2_1, test_mae_1, test_rmse_1, test_mape_1, _, _     = regression_metrics(df1['ytrue'], df1.iloc[:, i+5])
        test_mae_list1.append(test_mae_1)
        test_mape_list1.append(test_mape_1)

        test_r2_2, test_mae_2, test_rmse_2, test_mape_2, _, _     = regression_metrics(df2['ytrue'], df2.iloc[:, i+5])
        test_mae_list2.append(test_mae_2)
        test_mape_list2.append(test_mape_2)

        test_r2_3, test_mae_3, test_rmse_3, test_mape_3, _, _     = regression_metrics(df3['ytrue'], df3.iloc[:, i+5])
        test_mae_list3.append(test_mae_3)
        test_mape_list3.append(test_mape_3)
    
    results['weeks']      = Weeks
    results['Test_MAE_1']   = test_mae_list1
    results['Test_MAPE_1']  = test_mape_list1
    results['Test_MAE_2']   = test_mae_list2
    results['Test_MAPE_2']  = test_mape_list2
    results['Test_MAE_3']   = test_mae_list3
    results['Test_MAPE_3']  = test_mape_list3
 

    plt.rcParams["axes.grid"] = False
    fig, axs = plt.subplots(1, 2, figsize = (20, 5), sharex=True)

    axs[0].plot(results["weeks"], results["Test_MAE_1"],  "-*", color = 'black')
    axs[0].plot(results["weeks"], results["Test_MAE_2"],  "-o", color = 'blue')
    axs[0].plot(results["weeks"], results["Test_MAE_3"],  "-d", color = 'red')
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].set_ylabel('MAE (ton/ac)')
    
    axs[1].plot(results["weeks"], results["Test_MAPE_1"],  "-*", color = 'black', label = 'B1-2017')
    axs[1].plot(results["weeks"], results["Test_MAPE_2"],  "-o", color = 'blue', label = 'B2-2019')
    axs[1].plot(results["weeks"], results["Test_MAPE_3"],  "-d", color = 'red', label = 'B3-2018')
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].legend(loc='upper right')
    axs[1].set_ylabel('MAPE (%)')

def block_eval_barplot_S3(blocks_csv, cultivar_list = None, block_list = None):


    #blocks_csv = blocks_csv[(blocks_csv.block != 262016) & (blocks_csv.block != 262017)& (blocks_csv.block != 1932017)]
    #blocks_csv = blocks_csv[blocks_csv['cultivar'].isin(cultivar_list)]
    #blocks_csv = blocks_csv[blocks_csv['block'].isin(block_list)]
    blocks_csv = blocks_csv.sort_values(by = ['cultivar', 'block'])
    
    colors = ['#1f77b4', '#1f77b4','#1f77b4','#1f77b4','#1f77b4','#1f77b4','#1f77b4', 
                '#ff7f0e', '#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e',
                '#2ca02c', '#2ca02c','#2ca02c','#2ca02c', 
                '#d62728', '#d62728','#d62728', 
                '#9467bd',  '#9467bd', '#9467bd', '#9467bd', 
                '#8c564b', '#8c564b','#8c564b','#8c564b','#8c564b', '#8c564b','#8c564b','#8c564b',
                '#e377c2', '#e377c2','#e377c2','#e377c2', 
                '#7f7f7f', '#7f7f7f', '#7f7f7f']
    
    colours_labels = {"CS": "#1f77b4", "Ch": "#ff7f0e", "MB": "#2ca02c", "Me": "#d62728","MoA": "#9467bd", "Res": "#8c564b","Sym": "#e377c2", "Syr": "#7f7f7f"}


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (24, 8))


    plt.rcParams["axes.grid"] = False
    mae = blocks_csv.plot.bar(x = 'block', y = 'MAE', color = colors, width = 0.4, ax = ax1)
    ax1.legend([Patch(facecolor=colours_labels['CS']), Patch(facecolor=colours_labels['Ch']),
    Patch(facecolor=colours_labels['MB']),Patch(facecolor=colours_labels['Me']),Patch(facecolor=colours_labels['MoA']),Patch(facecolor=colours_labels['Res']),
    Patch(facecolor=colours_labels['Sym']), Patch(facecolor=colours_labels['Syr'])], ["CS", "Ch", "MB", "Me", "MoA", "Res", "Sym", "Syr"], loc='upper center', ncol=8, bbox_to_anchor=(0.5, 1.2))

    for i, v in enumerate(blocks_csv["MAE"].iteritems()):        
        ax1.text(i ,v[1], "{:.2f}".format(v[1]), color='k', position = (i-0.15, v[1] + 0.3), fontsize = 'x-small', rotation=90)
    ax1.set_ylim([0, blocks_csv['MAE'].max()+2])
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set(ylabel='MAE (ton/ac)')
    ax1.set_facecolor('white')
    ax1.axvline(x = 2.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax1.axvline(x = 6.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 10.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax1.axvline(x = 14.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax1.axvline(x = 18.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 22.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 25.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 29.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 33.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax1.axvline(x = 37.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 41.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    plt.setp(ax1.spines.values(), color='k')


    mape = blocks_csv.plot.bar(x = 'block', y = 'MAPE', color = colors, width = 0.4, ax = ax2)
    for j, m in enumerate(blocks_csv["MAPE"].iteritems()):        
        ax2.text(j ,m[1], "{:.2f}".format(m[1]), color='k', position = (j - 0.15, m[1] + 0.01), fontsize = 'x-small', rotation=90)
    ax2.set_ylim([0, blocks_csv['MAPE'].max()+0.1])
    ax2.set(xlabel = 'block', ylabel='MAPE (%)')
    ax2.set_facecolor('white')
    plt.setp(ax2.spines.values(), color='k')
    ax2.legend().set_visible(False)
    plt.xticks(rotation=90, fontsize = 14)
    ax2.axvline(x = 2.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax2.axvline(x = 6.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 10.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax2.axvline(x = 14.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax2.axvline(x = 18.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 22.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 25.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 29.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 33.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax2.axvline(x = 37.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 41.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    plt.show()
    None    

def concat_year_s2(df1, df2, df3):
    
    df = pd.DataFrame()
    tr_mae_m, v_mae_m, te_mae_m = [], [], []
    tr_mape_m, v_mape_m, te_mape_m = [], [], []
    tr_mae_s, v_mae_s, te_mae_s = [], [], []
    tr_mape_s, v_mape_s, te_mape_s = [], [], []
    weeks=[] 
    for i in range(15): 
        weeks.append(df1.iloc[i]['weeks'])
        tr_mae_m.append(np.mean([df1.iloc[i]['Train_MAE'], df2.iloc[i]['Train_MAE'], df3.iloc[i]['Train_MAE']]))
        tr_mae_s.append(np.std([df1.iloc[i]['Train_MAE'], df2.iloc[i]['Train_MAE'], df3.iloc[i]['Train_MAE']]))

        v_mae_m.append(np.mean([df1.iloc[i]['Valid_MAE'], df2.iloc[i]['Valid_MAE'], df3.iloc[i]['Valid_MAE']]))
        v_mae_s.append(np.std([df1.iloc[i]['Valid_MAE'], df2.iloc[i]['Valid_MAE'], df3.iloc[i]['Valid_MAE']]))

        te_mae_m.append(np.mean([df1.iloc[i]['Test_MAE'], df2.iloc[i]['Test_MAE'], df3.iloc[i]['Test_MAE']]))
        te_mae_s.append(np.std([df1.iloc[i]['Test_MAE'], df2.iloc[i]['Test_MAE'], df3.iloc[i]['Test_MAE']]))

        tr_mape_m.append(np.mean([df1.iloc[i]['Train_MAPE'], df2.iloc[i]['Train_MAPE'], df3.iloc[i]['Train_MAPE']]))
        tr_mape_s.append(np.std([df1.iloc[i]['Train_MAPE'], df2.iloc[i]['Train_MAPE'], df3.iloc[i]['Train_MAPE']]))

        v_mape_m.append(np.mean([df1.iloc[i]['Valid_MAPE'], df2.iloc[i]['Valid_MAPE'], df3.iloc[i]['Valid_MAPE']]))
        v_mape_s.append(np.std([df1.iloc[i]['Valid_MAPE'], df2.iloc[i]['Valid_MAPE'], df3.iloc[i]['Valid_MAPE']]))

        te_mape_m.append(np.mean([df1.iloc[i]['Test_MAPE'], df2.iloc[i]['Test_MAPE'], df3.iloc[i]['Test_MAPE']]))
        te_mape_s.append(np.std([df1.iloc[i]['Test_MAPE'], df2.iloc[i]['Test_MAPE'], df3.iloc[i]['Test_MAPE']]))

    df['weeks'] = weeks
    df['Train_MAE_M']  = tr_mae_m
    df['Train_MAE_S']  = tr_mae_s
    df['Valid_MAE_M']  = v_mae_m
    df['Valid_MAE_S']  = v_mae_s
    df['Test_MAE_M']  = te_mae_m
    df['Test_MAE_S']  = te_mae_s
    df['Train_MAPE_M'] = tr_mape_m
    df['Train_MAPE_S'] = tr_mape_s
    df['Valid_MAPE_M'] = v_mape_m
    df['Valid_MAPE_S'] = v_mape_s
    df['Test_MAPE_M']  = te_mape_m
    df['Test_MAPE_S']  = te_mape_s
    return df

def S2_mean_eval(df2016, df2017, df2018, df2019):
    
    df = pd.DataFrame()
    tr_mae_m, v_mae_m, te_mae_m = [], [], []
    tr_mape_m, v_mape_m, te_mape_m = [], [], []
    tr_mae_s, v_mae_s, te_mae_s = [], [], []
    tr_mape_s, v_mape_s, te_mape_s = [], [], []
    weeks=[] 
    for i in range(15): 
        weeks.append(df2016.iloc[i]['weeks'])
        tr_mae_m.append(np.mean([df2016.iloc[i]['Train_MAE_M'], df2017.iloc[i]['Train_MAE_M'], df2018.iloc[i]['Train_MAE_M'], df2019.iloc[i]['Train_MAE_M']]))
        tr_mae_s.append(np.std([df2016.iloc[i]['Train_MAE_M'], df2017.iloc[i]['Train_MAE_M'], df2018.iloc[i]['Train_MAE_M'], df2019.iloc[i]['Train_MAE_M']]))

        v_mae_m.append(np.mean([df2016.iloc[i]['Valid_MAE_M'], df2017.iloc[i]['Valid_MAE_M'], df2018.iloc[i]['Valid_MAE_M'], df2019.iloc[i]['Valid_MAE_M']]))
        v_mae_s.append(np.std([df2016.iloc[i]['Valid_MAE_M'], df2017.iloc[i]['Valid_MAE_M'], df2018.iloc[i]['Valid_MAE_M'], df2019.iloc[i]['Valid_MAE_M']]))

        te_mae_m.append(np.mean([df2016.iloc[i]['Test_MAE_M'], df2017.iloc[i]['Test_MAE_M'], df2018.iloc[i]['Test_MAE_M'], df2019.iloc[i]['Test_MAE_M']]))
        te_mae_s.append(np.std([df2016.iloc[i]['Test_MAE_M'], df2017.iloc[i]['Test_MAE_M'], df2018.iloc[i]['Test_MAE_M'], df2019.iloc[i]['Test_MAE_M']]))

        tr_mape_m.append(np.mean([df2016.iloc[i]['Train_MAPE_M'], df2017.iloc[i]['Train_MAPE_M'], df2018.iloc[i]['Train_MAPE_M'], df2019.iloc[i]['Train_MAPE_M']]))
        tr_mape_s.append(np.std([df2016.iloc[i]['Train_MAPE_M'], df2017.iloc[i]['Train_MAPE_M'], df2018.iloc[i]['Train_MAPE_M'], df2019.iloc[i]['Train_MAPE_M']]))

        v_mape_m.append(np.mean([df2016.iloc[i]['Valid_MAPE_M'], df2017.iloc[i]['Valid_MAPE_M'], df2018.iloc[i]['Valid_MAPE_M'], df2019.iloc[i]['Valid_MAPE_M']]))
        v_mape_s.append(np.std([df2016.iloc[i]['Valid_MAPE_M'], df2017.iloc[i]['Valid_MAPE_M'], df2018.iloc[i]['Valid_MAPE_M'], df2019.iloc[i]['Valid_MAPE_M']]))

        te_mape_m.append(np.mean([df2016.iloc[i]['Test_MAPE_M'], df2017.iloc[i]['Test_MAPE_M'], df2018.iloc[i]['Test_MAPE_M'], df2019.iloc[i]['Test_MAPE_M']]))
        te_mape_s.append(np.std([df2016.iloc[i]['Test_MAPE_M'], df2017.iloc[i]['Test_MAPE_M'], df2018.iloc[i]['Test_MAPE_M'], df2019.iloc[i]['Test_MAPE_M']]))

    df['weeks'] = weeks
    df['Train_MAE_M']  = tr_mae_m
    df['Train_MAE_S']  = tr_mae_s
    df['Valid_MAE_M']  = v_mae_m
    df['Valid_MAE_S']  = v_mae_s
    df['Test_MAE_M']  = te_mae_m
    df['Test_MAE_S']  = te_mae_s
    df['Train_MAPE_M'] = tr_mape_m
    df['Train_MAPE_S'] = tr_mape_s
    df['Valid_MAPE_M'] = v_mape_m
    df['Valid_MAPE_S'] = v_mape_s
    df['Test_MAPE_M']  = te_mape_m
    df['Test_MAPE_S']  = te_mape_s
    
    return df

def timeseries_10m_all(S1, S2_2016, S2_2017, S2_2018, S2_2019, S3): 
    Weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']

    #plt.rcParams["axes.grid"] = True
    fig, axs = plt.subplots(3, 2, figsize = (20, 15), sharex=True)


    #axs[0, 0].plot(S1["weeks"], S1["Train_MAE"], "-o")
    #axs[0, 0].plot(S1["weeks"], S1["Valid_MAE"], "-*")
    axs[0, 0].plot(S1["weeks"], S1["Test_MAE"]*2.2417,  "-d", color = 'k')
    axs[0, 0].set_ylabel('MAE (t/h), pixel-hold-out')
    axs[0, 0].set_facecolor('white')
    axs[0, 0].set_ylim([1.7, 8])
    plt.setp(axs[0, 0].spines.values(), color='k')
    
    #axs[0, 1].plot(S1["weeks"], S1["Train_MAPE"], "-o", label = 'train')
    #axs[0, 1].plot(S1["weeks"], S1["Valid_MAPE"], "-*", label = 'valid')
    axs[0, 1].plot(S1["weeks"], S1["Test_MAPE"]*100,  "-d", label = 'test', color = 'k')
    axs[0, 1].set_ylabel('MAPE (%)')
    axs[0, 1].set_facecolor('white')
    axs[0, 1].set_ylim([6.5, 30])
    plt.setp(axs[0, 1].spines.values(), color='k')
    axs[0, 1].legend(loc="upper right")
    
    #axs[1, 0].plot(S2["weeks"], S2["Train_MAE_M"], "-o")
    #axs[1, 0].fill_between(S2["weeks"], S2["Train_MAE_M"] - S2['Train_MAE_S'], S2["Train_MAE_M"]+ S2['Train_MAE_S'], alpha=.2)
    #axs[1, 0].plot(S2["weeks"], S2["Valid_MAE_M"], "-*")
    #axs[1, 0].fill_between(S2["weeks"], S2["Valid_MAE_M"] - S2['Valid_MAE_S'], S2["Valid_MAE_M"]+ S2['Valid_MAE_S'], alpha=.2)
    axs[1, 0].plot(S2_2016["weeks"], S2_2016["Test_MAE_M"]*2.2417,  "-d")
    axs[1, 0].fill_between(S2_2016["weeks"], S2_2016["Test_MAE_M"]*2.2417 - S2_2016['Test_MAE_S']*2.2417, S2_2016["Test_MAE_M"]*2.2417+ S2_2016['Test_MAE_S']*2.2417, alpha=.2)

    axs[1, 0].plot(S2_2017["weeks"], S2_2017["Test_MAE_M"]*2.2417,  "-d")
    axs[1, 0].fill_between(S2_2017["weeks"], S2_2017["Test_MAE_M"]*2.2417 - S2_2017['Test_MAE_S']*2.2417, S2_2017["Test_MAE_M"]*2.2417+ S2_2017['Test_MAE_S']*2.2417, alpha=.2)

    axs[1, 0].plot(S2_2018["weeks"], S2_2018["Test_MAE_M"]*2.2417,  "-d")
    axs[1, 0].fill_between(S2_2018["weeks"], S2_2018["Test_MAE_M"]*2.2417 - S2_2018['Test_MAE_S']*2.2417, S2_2018["Test_MAE_M"]*2.2417+ S2_2018['Test_MAE_S']*2.2417, alpha=.2)

    axs[1, 0].plot(S2_2019["weeks"], S2_2019["Test_MAE_M"]*2.2417,  "-d")
    axs[1, 0].fill_between(S2_2019["weeks"], S2_2019["Test_MAE_M"]*2.2417 - S2_2019['Test_MAE_S']*2.2417, S2_2019["Test_MAE_M"]*2.2417+ S2_2019['Test_MAE_S']*2.2417, alpha=.2)
    axs[1, 0].set_ylabel('MAE (t/h), year-hold-out')
    axs[1, 0].set_facecolor('white')
    axs[1, 0].set_ylim([1.7, 8])
    plt.setp(axs[1, 0].spines.values(), color='k')
    
    #axs[1, 1].plot(S2["weeks"], S2["Train_MAPE_M"], "-o")
    #axs[1, 1].fill_between(S2["weeks"], S2["Train_MAPE_M"] - S2['Train_MAPE_S'], S2["Train_MAPE_M"]+ S2['Train_MAPE_S'], alpha=.2)
    #axs[1, 1].plot(S2["weeks"], S2["Valid_MAPE_M"], "-*")
    #axs[1, 1].fill_between(S2["weeks"], S2["Valid_MAPE_M"] - S2['Valid_MAPE_S'], S2["Valid_MAPE_M"]+ S2['Valid_MAPE_S'], alpha=.2)
    axs[1, 1].plot(S2_2016["weeks"], S2_2016["Test_MAPE_M"]*100,  "-d", label = 'test_2016')
    axs[1, 1].fill_between(S2_2016["weeks"], S2_2016["Test_MAPE_M"]*100 - S2_2016['Test_MAPE_S']*100, S2_2016["Test_MAPE_M"]*100+ S2_2016['Test_MAPE_S']*100, alpha=.2)

    axs[1, 1].plot(S2_2017["weeks"], S2_2017["Test_MAPE_M"]*100,  "-d", label = 'test_2017')
    axs[1, 1].fill_between(S2_2017["weeks"], S2_2017["Test_MAPE_M"]*100 - S2_2017['Test_MAPE_S']*100, S2_2017["Test_MAPE_M"]*100+ S2_2017['Test_MAPE_S']*100, alpha=.2)

    axs[1, 1].plot(S2_2018["weeks"], S2_2018["Test_MAPE_M"]*100,  "-d", label = 'test_2018')
    axs[1, 1].fill_between(S2_2018["weeks"], S2_2018["Test_MAPE_M"]*100 - S2_2018['Test_MAPE_S']*100, S2_2018["Test_MAPE_M"]*100+ S2_2018['Test_MAPE_S']*100, alpha=.2)

    axs[1, 1].plot(S2_2019["weeks"], S2_2019["Test_MAPE_M"]*100,  "-d", label = 'test_2019')
    axs[1, 1].fill_between(S2_2019["weeks"], S2_2019["Test_MAPE_M"]*100 - S2_2019['Test_MAPE_S']*100, S2_2019["Test_MAPE_M"]*100+ S2_2019['Test_MAPE_S']*100, alpha=.2)


    axs[1, 1].set_ylabel('MAPE (%)')
    axs[1, 1].set_ylim([6.5, 30])
    axs[1, 1].set_facecolor('white')
    axs[1, 1].legend(loc="upper right", ncol=4)
    plt.setp(axs[1, 1].spines.values(), color='k')

    #axs[2, 0].plot(S3["weeks"], S3["Train_MAE"], "-o")
    #axs[2, 0].plot(S3["weeks"], S3["Valid_MAE"], "-*")
    axs[2, 0].plot(S3["weeks"], S3["Test_MAE"]*2.2417,  "-d", color = 'k')
    axs[2, 0].set_ylabel('MAE (t/h), block-hold-out')
    axs[2, 0].set_facecolor('white')
    axs[2, 0].tick_params(axis='x', rotation=45)
    axs[2, 0].set_ylim([1.7, 8])
    plt.setp(axs[2, 0].spines.values(), color='k')
    
    #axs[2, 1].plot(S3["weeks"], S3["Train_MAPE"], "-o")
    #axs[2, 1].plot(S3["weeks"], S3["Valid_MAPE"], "-*")
    axs[2, 1].plot(S3["weeks"], S3["Test_MAPE"]*100,  "-d", label = 'test', color = 'k')
    axs[2, 1].set_ylabel('MAPE (%)')
    axs[2, 1].set_facecolor('white')
    axs[2, 1].tick_params(axis='x', rotation=45)
    axs[2, 1].set_ylim([6.5, 30])
    plt.setp(axs[2, 1].spines.values(), color='k')
    axs[2, 1].legend(loc="upper right")

def yield_true_pred_plot(ytrue, ypred, min_v = None, max_v= None):

    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.rcParams["axes.grid"] = False
    fig, axs = plt.subplots(1, 4, figsize = (24, 8))

    img1 = axs[0].imshow(ytrue)
    axs[0].set_title('Yield Observation', fontsize = 14)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar1 = fig.colorbar(img1,  cax=cax)
    img1.set_clim(min_v, max_v)

    img2 = axs[1].imshow(ypred)
    axs[1].set_title('Yield Prediction', fontsize = 14)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar2 =fig.colorbar(img2, cax=cax)
    img2.set_clim(min_v, max_v)
    axs[1].get_yaxis().set_visible(False)

    #mae_map = image_subtract(ytrue, ypred) 
    #mape_map = image_subtract(ytrue, ypred) 
    mae_map, mape_map = image_mae_mape_map(ytrue, ypred)
    img3 = axs[2].imshow(mae_map, cmap = 'viridis') #, cmap = 'magma'
    #xlabel_text = (r"($PSNR = {:.2f}$" + ", " + r"$SSIM = {:.2f}$)").format(PSNR, ssim_value)
    axs[2].set_title('MAE Map (t/h)', fontsize = 14)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar2 =fig.colorbar(img3, cax=cax)
    img3.set_clim(-1, np.max(mae_map))
    axs[2].get_yaxis().set_visible(False)
    #axs[2].get_xaxis().set_visible(False)
    #axs[2].set_xlabel(xlabel_text)

    img4 = axs[3].imshow(mape_map, cmap = 'viridis') #
    #xlabel_text = (r"($PSNR = {:.2f}$" + ", " + r"$SSIM = {:.2f}$)").format(PSNR, ssim_value)
    axs[3].set_title('MAPE Map (%)', fontsize = 14)
    divider = make_axes_locatable(axs[3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar3 =fig.colorbar(img4, cax=cax)
    img4.set_clim(-5, 20)
    axs[3].get_yaxis().set_visible(False)

    #return mape_map
    #plt.savefig('./imgs/B186_1m.png', dpi = 300)

def image_mae_mape_map(ytrue, ypred): 

    w = ytrue.shape[0]
    h = ytrue.shape[1] 

    ytrue_flat = ytrue.ravel()
    ypred_flat = ypred.ravel()
    out_mae    = np.empty_like(ytrue_flat)
    out_mape   = np.empty_like(ytrue_flat)

    for i in range(len(ytrue_flat)):
        if (ytrue_flat[i] == -1) and (ypred_flat[i] == -1):
            out_mae[i]  = -10
            out_mape[i] = -10
        elif abs(ytrue_flat[i] - ypred_flat[i]) > 5:
            out_mae[i]  = -10
            out_mape[i] = -10
        else: 
            out_mae[i]  = abs(ytrue_flat[i] - ypred_flat[i])
            out_mape[i] = ((abs(ytrue_flat[i] - ypred_flat[i]))/ytrue_flat[i])*100

    out1 = out_mae.reshape(ytrue.shape[0], ytrue.shape[1])
    out2 = out_mape.reshape(ytrue.shape[0], ytrue.shape[1])

    return out1, out2

def block_true_pred_mtx(df, block_id, aggregation = None, spatial_resolution  = None, scale = None):
    
    name_split = os.path.split(str(block_id))[-1]
    root_name  = name_split.replace(name_split[-4:], '')
    year       = name_split[-4:]
    
    if len(root_name) == 1:
        this_block_name = 'LIV_00' + root_name + '_' + year
    elif len(root_name) == 2:
        this_block_name = 'LIV_0' + root_name + '_' + year
    elif len(root_name) == 3:
        this_block_name = 'LIV_' + root_name + '_' + year
    
    #print(this_block_name)
    blocks_df = df.groupby(by = 'block')
    this_block_df = blocks_df.get_group(block_id)
    
    
    
    res           = {key: blocks_size[key] for key in blocks_size.keys() & {this_block_name}}
    list_d        = res.get(this_block_name)
    block_x_size  = int(list_d[0]/spatial_resolution)
    block_y_size  = int(list_d[1]/spatial_resolution)

    pred_out = np.full((block_x_size, block_y_size), -1) 
    true_out = np.full((block_x_size, block_y_size), -1)  

    for x in range(block_x_size):
        for y in range(block_y_size):
            
            new            = this_block_df.loc[(this_block_df['x'] == x)&(this_block_df['y'] == y)] 
            if len(new) > 0:
                pred_out[x, y] = new['ypred_w15'].min()*2.2417
                true_out[x, y] = new['ytrue'].mean()*2.2417


    if aggregation is True: 
        
        print(f"{pred_out.shape}|{true_out.shape}")
        pred_agg = aggregate2(pred_out, scale)
        true_agg = aggregate2(true_out, scale)
        print(f"Agg: {pred_agg.shape}|{true_agg.shape}")
        
        df_agg = pd.DataFrame() 
        
        flat_pred_agg = pred_agg.flatten()
        flat_pred_agg = flat_pred_agg[flat_pred_agg != -1]
        
        flat_true_agg = true_agg.flatten()
        flat_true_agg = flat_true_agg[flat_true_agg != -1] 
              
        df_agg['ytrue'] = flat_true_agg
        df_agg['ypred'] = flat_pred_agg 
        
        return df_agg, true_agg, pred_agg
    else:
        return this_block_df, true_out, pred_out            

def ScenarioEvaluation2D(pred_npy):

    OutDF = pd.DataFrame()
    out_ytrue= []
    out_ypred_w1, out_ypred_w2, out_ypred_w3, out_ypred_w4, out_ypred_w5, out_ypred_w6, out_ypred_w7, out_ypred_w8 = [],[],[],[],[],[],[],[]
    out_ypred_w9, out_ypred_w10, out_ypred_w11, out_ypred_w12, out_ypred_w13, out_ypred_w14, out_ypred_w15 = [],[],[],[],[],[],[]
  
    for l in range(len(pred_npy)):

        block_key = next(iter(pred_npy[l]))
        block_allpatch = pred_npy[l][block_key]        
            
        ytrue     = pred_npy[l]['ytrue']
        ytrue_flat = ytrue.flatten()
        out_ytrue.append(ytrue_flat)
        
        ypred_w1  = pred_npy[l]['ypred_w1']
        ypred_w1_flat = ypred_w1.flatten()
        out_ypred_w1.append(ypred_w1_flat)
        
        ypred_w2  = pred_npy[l]['ypred_w2']
        ypred_w2_flat = ypred_w2.flatten()
        out_ypred_w2.append(ypred_w2_flat)
        
        ypred_w3  = pred_npy[l]['ypred_w3']
        ypred_w3_flat = ypred_w3.flatten()
        out_ypred_w3.append(ypred_w3_flat)
        
        ypred_w4  = pred_npy[l]['ypred_w4']
        ypred_w4_flat = ypred_w4.flatten()
        out_ypred_w4.append(ypred_w4_flat)
        
        ypred_w5  = pred_npy[l]['ypred_w5']
        ypred_w5_flat = ypred_w5.flatten()
        out_ypred_w5.append(ypred_w5_flat)
        
        ypred_w6  = pred_npy[l]['ypred_w6']
        ypred_w6_flat = ypred_w6.flatten()
        out_ypred_w6.append(ypred_w6_flat)
        
        ypred_w7  = pred_npy[l]['ypred_w7']
        ypred_w7_flat = ypred_w7.flatten()
        out_ypred_w7.append(ypred_w7_flat)
        
        ypred_w8  = pred_npy[l]['ypred_w8']
        ypred_w8_flat = ypred_w8.flatten()
        out_ypred_w8.append(ypred_w8_flat)
        
        ypred_w9  = pred_npy[l]['ypred_w9']
        ypred_w9_flat = ypred_w9.flatten()
        out_ypred_w9.append(ypred_w9_flat)
        
        ypred_w10  = pred_npy[l]['ypred_w10']
        ypred_w10_flat = ypred_w10.flatten()
        out_ypred_w10.append(ypred_w10_flat)
        
        ypred_w11  = pred_npy[l]['ypred_w11']
        ypred_w11_flat = ypred_w11.flatten()
        out_ypred_w11.append(ypred_w11_flat)
        
        ypred_w12  = pred_npy[l]['ypred_w12']
        ypred_w12_flat = ypred_w12.flatten()
        out_ypred_w12.append(ypred_w12_flat)
        
        ypred_w13  = pred_npy[l]['ypred_w13']
        ypred_w13_flat = ypred_w13.flatten()
        out_ypred_w13.append(ypred_w13_flat)
        
        ypred_w14  = pred_npy[l]['ypred_w14']
        ypred_w14_flat = ypred_w14.flatten()
        out_ypred_w14.append(ypred_w14_flat)
        
        ypred_w15  = pred_npy[l]['ypred_w15']
        ypred_w15_flat = ypred_w15.flatten()
        out_ypred_w15.append(ypred_w15_flat)
      


            
    out_ytrue = np.concatenate(out_ytrue)#.astype(np.float16)
    out_ypred_w1 = np.concatenate(out_ypred_w1) 
    out_ypred_w2 = np.concatenate(out_ypred_w2) 
    out_ypred_w3 = np.concatenate(out_ypred_w3) 
    out_ypred_w4 = np.concatenate(out_ypred_w4) 
    out_ypred_w5 = np.concatenate(out_ypred_w5) 
    out_ypred_w6 = np.concatenate(out_ypred_w6) 
    out_ypred_w7 = np.concatenate(out_ypred_w7) 
    out_ypred_w8 = np.concatenate(out_ypred_w8) 
    out_ypred_w9 = np.concatenate(out_ypred_w9) 
    out_ypred_w10 = np.concatenate(out_ypred_w10) 
    out_ypred_w11 = np.concatenate(out_ypred_w11) 
    out_ypred_w12 = np.concatenate(out_ypred_w12) 
    out_ypred_w13 = np.concatenate(out_ypred_w13) 
    out_ypred_w14 = np.concatenate(out_ypred_w14) 
    out_ypred_w15 = np.concatenate(out_ypred_w15) 

    
    OutDF['ytrue'] = out_ytrue
    OutDF['ypred_w1'] = out_ypred_w1
    OutDF['ypred_w2'] = out_ypred_w2 
    OutDF['ypred_w3'] = out_ypred_w3 
    OutDF['ypred_w4'] = out_ypred_w4 
    OutDF['ypred_w5'] = out_ypred_w5 
    OutDF['ypred_w6'] = out_ypred_w6 
    OutDF['ypred_w7'] = out_ypred_w7 
    OutDF['ypred_w8'] = out_ypred_w8 
    OutDF['ypred_w9'] = out_ypred_w9 
    OutDF['ypred_w10'] = out_ypred_w10 
    OutDF['ypred_w11'] = out_ypred_w11 
    OutDF['ypred_w12'] = out_ypred_w12 
    OutDF['ypred_w13'] = out_ypred_w13 
    OutDF['ypred_w14'] = out_ypred_w14 
    OutDF['ypred_w15'] = out_ypred_w15


    return OutDF

def block_true_pred_mtx_timeseries(df, block_id, aggregation = None, spatial_resolution  = None):
    
    name_split = os.path.split(str(block_id))[-1]
    root_name  = name_split.replace(name_split[-4:], '')
    year       = name_split[-4:]
    
    if len(root_name) == 1:
        this_block_name = 'LIV_00' + root_name + '_' + year
    elif len(root_name) == 2:
        this_block_name = 'LIV_0' + root_name + '_' + year
    elif len(root_name) == 3:
        this_block_name = 'LIV_' + root_name + '_' + year
    
    #print(this_block_name)
    blocks_df = df.groupby(by = 'block')
    this_block_df = blocks_df.get_group(block_id)
    
    
    res           = {key: blocks_size[key] for key in blocks_size.keys() & {this_block_name}}
    list_d        = res.get(this_block_name)
    block_x_size  = int(list_d[0]/spatial_resolution)
    block_y_size  = int(list_d[1]/spatial_resolution)
    
    print(this_block_df.shape)
    true_out = np.full((block_x_size, block_y_size), -1)
    pred_out1 = np.full((block_x_size, block_y_size), -1)   
    pred_out2 = np.full((block_x_size, block_y_size), -1) 
    pred_out3 = np.full((block_x_size, block_y_size), -1) 
    pred_out4 = np.full((block_x_size, block_y_size), -1) 
    pred_out5 = np.full((block_x_size, block_y_size), -1) 
    pred_out6 = np.full((block_x_size, block_y_size), -1) 
    pred_out7 = np.full((block_x_size, block_y_size), -1)
    pred_out8 = np.full((block_x_size, block_y_size), -1) 
    pred_out9 = np.full((block_x_size, block_y_size), -1) 
    pred_out10 = np.full((block_x_size, block_y_size), -1) 
    pred_out11 = np.full((block_x_size, block_y_size), -1) 
    pred_out12 = np.full((block_x_size, block_y_size), -1) 
    pred_out13 = np.full((block_x_size, block_y_size), -1) 
    pred_out14 = np.full((block_x_size, block_y_size), -1) 
    pred_out15 = np.full((block_x_size, block_y_size), -1) 


    for x in range(block_x_size):
        for y in range(block_y_size):
            
            new            = this_block_df.loc[(this_block_df['x'] == x)&(this_block_df['y'] == y)] 

            if len(new) > 0:
                true_out[x, y] = new['ytrue'].mean()

                pred_out1[x, y] = new['ypred_w1'].max()
                pred_out2[x, y] = new['ypred_w2'].max()
                pred_out3[x, y] = new['ypred_w3'].max()
                pred_out4[x, y] = new['ypred_w4'].max()
                pred_out5[x, y] = new['ypred_w5'].max()
                pred_out6[x, y] = new['ypred_w6'].max()
                pred_out7[x, y] = new['ypred_w7'].max()
                pred_out8[x, y] = new['ypred_w8'].max()
                pred_out9[x, y] = new['ypred_w9'].max()
                pred_out10[x, y] = new['ypred_w10'].max()
                pred_out11[x, y] = new['ypred_w11'].max()
                pred_out12[x, y] = new['ypred_w12'].max()
                pred_out13[x, y] = new['ypred_w13'].max()
                pred_out14[x, y] = new['ypred_w14'].max()
                pred_out15[x, y] = new['ypred_w15'].max()



        #from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.rcParams["axes.grid"] = False
    fig, axs = plt.subplots(4, 4, figsize = (24, 24))

    img1 = axs[0, 0].imshow(true_out)
    axs[0, 0].set_title('Yield Observation', fontsize = 16)
    divider = make_axes_locatable(axs[0, 0])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar1 = fig.colorbar(img1,  cax=cax)
    img1.set_clim(5, 20)

    img2 = axs[0, 1].imshow(pred_out15)
    tr_r2_1, tr_mae_1, tr_rmse_1, tr_mape_1, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w1'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_1, tr_mape_1)
    axs[0, 1].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[0, 1].set_title('Yield Prediction Week 1', fontsize = 16)
    divider = make_axes_locatable(axs[0, 1])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar2 =fig.colorbar(img2, cax=cax)
    img2.set_clim(5, 20)
    axs[0, 1].get_yaxis().set_visible(False)


    img3 = axs[0, 2].imshow(pred_out2)
    tr_r2_2, tr_mae_2, tr_rmse_2, tr_mape_2, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w2'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_2, tr_mape_2)
    axs[0, 2].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[0, 2].set_title('Yield Prediction Week 2', fontsize = 16)
    divider = make_axes_locatable(axs[0, 2])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar3 =fig.colorbar(img3, cax=cax)
    img3.set_clim(5, 20)
    axs[0, 2].get_yaxis().set_visible(False)


    img4 = axs[0, 3].imshow(pred_out3)
    tr_r2_3, tr_mae_3, tr_rmse_3, tr_mape_3, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w3'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_3, tr_mape_3)
    axs[0, 3].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[0, 3].set_title('Yield Prediction Week 3', fontsize = 16)
    divider = make_axes_locatable(axs[0, 3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar4 =fig.colorbar(img4, cax=cax)
    img4.set_clim(5, 20)
    axs[0, 3].get_yaxis().set_visible(False)


    img5 = axs[1, 0].imshow(pred_out4)
    tr_r2_4, tr_mae_4, tr_rmse_4, tr_mape_4, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w4'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_4, tr_mape_4)
    axs[1, 0].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[1, 0].set_title('Yield Prediction Week 4', fontsize = 16)
    divider = make_axes_locatable(axs[1, 0])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar5 =fig.colorbar(img5, cax=cax)
    img5.set_clim(5, 20)


    img6 = axs[1, 1].imshow(pred_out5)
    tr_r2_5, tr_mae_5, tr_rmse_5, tr_mape_5, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w5'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_5, tr_mape_5)
    axs[1, 1].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[1, 1].set_title('Yield Prediction Week 5', fontsize = 16)
    divider = make_axes_locatable(axs[1, 1])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar6 =fig.colorbar(img6, cax=cax)
    img6.set_clim(5, 20)
    axs[1, 1].get_yaxis().set_visible(False)



    img7 = axs[1, 2].imshow(pred_out6)
    tr_r2_6, tr_mae_6, tr_rmse_6, tr_mape_6, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w6'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_6, tr_mape_6)
    axs[1, 2].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[1, 2].set_title('Yield Prediction Week 6', fontsize = 16)
    divider = make_axes_locatable(axs[1, 2])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar7 =fig.colorbar(img7, cax=cax)
    img7.set_clim(5, 20)
    axs[1, 2].get_yaxis().set_visible(False)



    img8 = axs[1, 3].imshow(pred_out7)
    tr_r2_7, tr_mae_7, tr_rmse_7, tr_mape_7, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w7'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_7, tr_mape_7)
    axs[1, 3].legend([extra], [scores], loc='upper left', fontsize = 16)
    axs[1, 3].set_title('Yield Prediction Week 7', fontsize = 16)
    divider = make_axes_locatable(axs[1, 3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar8 =fig.colorbar(img8, cax=cax)
    img8.set_clim(5, 20)
    axs[1, 3].get_yaxis().set_visible(False)


    img9 = axs[2, 0].imshow(pred_out8)
    tr_r2_8, tr_mae_8, tr_rmse_8, tr_mape_8, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w8'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_8, tr_mape_8)
    axs[2, 0].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[2, 0].set_title('Yield Prediction Week 8', fontsize = 14)
    #divider = make_axes_locatable(axs[2, 0])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar9 =fig.colorbar(img9, cax=cax)
    img9.set_clim(5, 20)




    img10 = axs[2, 1].imshow(pred_out9)
    tr_r2_9, tr_mae_9, tr_rmse_9, tr_mape_9, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w9'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_9, tr_mape_9)
    axs[2, 1].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[2, 1].set_title('Yield Prediction Week 9', fontsize = 16)
    #divider = make_axes_locatable(axs[2, 1])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar10 =fig.colorbar(img10, cax=cax)
    img10.set_clim(5, 20)
    axs[2, 1].get_yaxis().set_visible(False)



    img11 = axs[2, 2].imshow(pred_out10)
    tr_r2_10, tr_mae_10, tr_rmse_10, tr_mape_10, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w10'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}'+ '\n' + r'$MAPE={:.2f}$').format(tr_mae_10, tr_mape_10)
    axs[2, 2].legend([extra], [scores], loc='upper left', fontsize = 16)
    axs[2, 2].set_title('Yield Prediction Week 10', fontsize = 16)
    #divider = make_axes_locatable(axs[2, 2])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar11 =fig.colorbar(img11, cax=cax)
    img11.set_clim(5, 20)
    axs[2, 2].get_yaxis().set_visible(False)



    img12 = axs[2, 3].imshow(pred_out11)
    tr_r2_11, tr_mae_11, tr_rmse_11, tr_mape_11, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w11'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_11, tr_mape_11)
    axs[2, 3].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[2, 3].set_title('Yield Prediction Week 11', fontsize = 16)
    divider = make_axes_locatable(axs[2, 3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar12 =fig.colorbar(img12, cax=cax)
    img12.set_clim(5, 20)
    axs[2, 3].get_yaxis().set_visible(False)



    img13 = axs[3, 0].imshow(pred_out12)
    tr_r2_12, tr_mae_12, tr_rmse_12, tr_mape_12, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w12'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_12, tr_mape_12)
    axs[3, 0].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[3, 0].set_title('Yield Prediction Week 12', fontsize = 16)
    #divider = make_axes_locatable(axs[3, 0])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar13 =fig.colorbar(img13, cax=cax)
    img13.set_clim(5, 20)



    img14 = axs[3, 1].imshow(pred_out13)
    tr_r2_13, tr_mae_13, tr_rmse_13, tr_mape_13, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w13'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_13, tr_mape_13)
    axs[3, 1].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[3, 1].set_title('Yield Prediction Week 13', fontsize = 16)
    #divider = make_axes_locatable(axs[3, 1])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar14 =fig.colorbar(img14, cax=cax)
    img14.set_clim(5, 20)
    axs[3, 1].get_yaxis().set_visible(False)


    img15 = axs[3, 2].imshow(pred_out14)
    tr_r2_14, tr_mae_14, tr_rmse_14, tr_mape_14, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w14'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_14, tr_mape_14)
    axs[3, 2].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[3, 2].set_title('Yield Prediction Week 14', fontsize = 16)
    #divider = make_axes_locatable(axs[3, 2])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar15 =fig.colorbar(img15, cax=cax)
    img15.set_clim(5, 20)
    axs[3, 2].get_yaxis().set_visible(False)



    img16 = axs[3, 3].imshow(pred_out15)
    tr_r2_15, tr_mae_15, tr_rmse_15, tr_mape_15, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w15'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_15, tr_mape_15)
    axs[3, 3].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[3, 3].set_title('Yield Prediction Week 15', fontsize = 16)
    divider = make_axes_locatable(axs[3, 3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar16 =fig.colorbar(img16, cax=cax)
    img16.set_clim(5, 20)
    axs[3, 3].get_yaxis().set_visible(False)

    fig.tight_layout()
    return this_block_df 

def scenarios_satterplot(train_df, valid_df, test_df, week = None, cmap  = None, mincnt = None):

    week_pred = 'ypred_w' + str(week)

    w_train_e1  = train_df[['ytrue', week_pred]]
    w_train_e1  = w_train_e1.rename(columns={week_pred: "ypred"})
    tarin_true  = w_train_e1['ytrue']
    train_pred  = w_train_e1['ypred']
    tr_r2, tr_mae, tr_rmse, tr_mape, _,_ = regression_metrics(tarin_true, train_pred)

    TR = sns.jointplot(x=tarin_true, y=train_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = cmap , mincnt=mincnt, joint_kws={"facecolor": 'white'}, marginal_kws = {"ymax": 50000}) # , , 
    for patch in TR.ax_marg_x.patches:
        patch.set_facecolor('grey')

    #for patch in TR.ax_marg_x.patches:
     #   patch.set_clim(0, 50000)

    for patch in TR.ax_marg_y.patches:
        patch.set_facecolor('grey')



    #TR.ax_marg_y.patches.ylim(0, 50000)
    TR.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)

    plt.xlabel('Measured (ton/ac), S1')
    plt.ylabel('Predict (ton/ac)')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(tr_r2, tr_mae, tr_rmse, tr_mape)
    plt.legend([extra], [scores], loc='upper left')
    #plt.title('Train Data')
    #========================================================
    w_valid_e1  = valid_df[['ytrue', week_pred]]
    w_valid_e1  = w_valid_e1.rename(columns={week_pred: "ypred"})
    valid_true = w_valid_e1['ytrue']
    valid_pred = w_valid_e1['ypred']
    val_r2, val_mae, val_rmse, val_mape, _,_ = regression_metrics(valid_true, valid_pred)

    Va = sns.jointplot(x = valid_true, y = valid_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = 'viridis', mincnt = 4000) #palette ='flare', ,   
    for patch in Va.ax_marg_x.patches:
        patch.set_facecolor('grey')
    for patch in Va.ax_marg_y.patches:
        patch.set_facecolor('grey')


    Va.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)
    plt.xlabel('Measured (ton/ac), S2')
    plt.ylabel('')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(val_r2, val_mae, val_rmse, val_mape)
    plt.legend([extra], [scores], loc='upper left')
    #plt.title('Validation Data')
    #========================================================
    w_test_e1  = test_df[['ytrue', week_pred]]
    w_test_e1  = w_test_e1.rename(columns={week_pred: "ypred"})
    test_true = w_test_e1['ytrue']
    valid_pred = w_test_e1['ypred']
    test_r2, test_mae, test_rmse, test_mape, _,_ = regression_metrics(test_true, valid_pred)

    Te = sns.jointplot(x=test_true, y = valid_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = cmap, mincnt = mincnt) #palette ='flare', ,  
    for patch in Te.ax_marg_x.patches:
        patch.set_facecolor('grey')
    for patch in Te.ax_marg_y.patches:
        patch.set_facecolor('grey') 

    Te.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)

    plt.xlabel('Measured (ton/ac), S3')
    plt.ylabel('')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(test_r2, test_mae, test_rmse, test_mape)
    plt.legend([extra], [scores], loc='upper left')



    fig = plt.figure(figsize=(21,7))
    gs = gridspec.GridSpec(1, 3)

    mg0 = SeabornFig2Grid(TR, fig, gs[0])
    mg1 = SeabornFig2Grid(Va, fig, gs[1])
    mg2 = SeabornFig2Grid(Te, fig, gs[2])


    gs.tight_layout(fig)
    #gs.update(top=0.7)

    plt.show()