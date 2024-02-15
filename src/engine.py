import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"

from src import configs
#======================================================================================================================================#
#====================================================== Training Config ===============================================================#
#======================================================================================================================================#   
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

seed = 1987 + get_rank()
torch.manual_seed(seed)
np.random.seed(seed)

def mse_loss(inputs, targets):
    loss = (inputs - targets) ** 2
    loss = torch.mean(loss)
    
    return loss


class EarlyStopping():
    def __init__(self, tolerance=30, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, status):

        if status is True:
            self.counter = 0
        elif status is False: 
            self.counter +=1

        print(f"count: {self.counter}")
        if self.counter >= self.tolerance:  
                self.early_stop = True

def save_loss_df(loss_stat, loss_df_name, loss_fig_name):

    df = pd.DataFrame.from_dict(loss_stat).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    df.to_csv(loss_df_name) 
    plt.figure(figsize=(12,8))
    sns.lineplot(data=df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.ylim(0, df['value'].max())
    plt.savefig(loss_fig_name, dpi = 300)

def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves a model checkpoint during training.

    Parameters:
    - state (dict): State to save, including model and optimizer states.
    - filename (str): File name to save the checkpoint.
    """
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Loads a checkpoint into a model and optimizer.

    Parameters:
    - checkpoint_path (str): Path to the checkpoint file.
    - model (nn.Module): Model to load the checkpoint into.
    - optimizer (torch.optim): Optimizer to load the checkpoint into.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'], checkpoint['best_val_loss']

class YieldEst:
    """
    This class implements the training and evaluation of a neural network model for yield estimation.

    Attributes:
        model (nn.Module): The neural network model.
        lr (float): Learning rate for the optimizer.
        wd (float): Weight decay for regularization.
        exp (str): Experiment identifier used for file naming and directory structuring.
        optimizer (torch.optim): Optimizer for training the model.
        exp_output_dir (str): Directory for experiment outputs.
        best_model_name (str): Path for saving the best model.
        checkpoint_dir (str): Directory for saving checkpoints.
        loss_fig_name (str): File name for the loss figure.
        loss_df_name (str): File name for the loss data in CSV format.
        train_df_name (str): File name for the training data.
        valid_df_name (str): File name for the validation data.
        test_df_name (str): File name for the test data.
        timeseries_fig (str): File name for the time series figure.
        scatterplot (str): File name for the scatter plot.

    Methods:
        train: Trains the model using the provided data loaders.
        predict: Generates predictions using the trained model.
        _return_pred_df: Helper function to process prediction results into a DataFrame.
        xy_vector_generator: Generates X and Y coordinate vectors for a given point and window size.
    """

    def __init__(self, model, lr: float, wd: float, exp: str):
        # Initialize model parameters and directories
        self.model = model
        self.lr = lr
        self.wd = wd
        self.exp = exp

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)

        self.exp_output_dir = '/data2/hkaman/Imbalance/EXPs/CNNs/' + 'EXP_' + self.exp

        self.best_model_name = os.path.join(self.exp_output_dir, 'best_model_' + self.exp + '.pth')
        self.best_checkpoint_dir = os.path.join(self.exp_output_dir, 'best_checkpoints_' + self.exp + '.pth')
        self.checkpoint_dir = os.path.join(self.exp_output_dir, 'checkpoints')
        self.loss_fig_name = os.path.join(self.exp_output_dir, 'loss', 'loss_' + self.exp + '.png')
        self.loss_df_name = os.path.join(self.exp_output_dir, 'loss', 'loss_' + self.exp + '.csv')
        self.train_df_name = os.path.join(self.exp_output_dir, self.exp + '_train.csv')
        self.valid_df_name = os.path.join(self.exp_output_dir, self.exp + '_valid.csv')
        self.test_df_name = os.path.join(self.exp_output_dir, self.exp + '_test.csv')
        self.timeseries_fig = os.path.join(self.exp_output_dir, self.exp + '_timeseries.png')
        self.scatterplot = os.path.join(self.exp_output_dir, self.exp + '_scatterplot.png')


    def train(self, data_loader_training, data_loader_validate, epochs: int, loss_stop_tolerance: int):
        """
        Trains the model using the provided training and validation data loaders.

        Parameters:
        - data_loader_training: DataLoader for the training dataset.
        - data_loader_validate: DataLoader for the validation dataset.
        - loss_type (str): Type of loss function to use ('mse', 'wmse', 'huber', 'wass').
        - epochs (int): Number of epochs to train the model.
        - loss_stop_tolerance (int): Early stopping tolerance level.
        """

        best_val_loss  = 1e15 # initial dummy value
        early_stopping = EarlyStopping(tolerance = loss_stop_tolerance, min_delta = 50)
        loss_stats = {'train': [],"val": []}
        

        for epoch in range(1, epochs + 1):

            training_start_time = time.time()
            train_epoch_loss = 0
            self.model.train()

            for batch, sample in enumerate(data_loader_training):
                
                xtrain = sample['image'].to(device)
                ytrain_true = sample['mask'][:,:,:,:,0].to(device)
                embmatrix_train = sample['EmbMatrix'].to(device)

                list_ytrain_pred  = self.model(xtrain, embmatrix_train)
                self.optimizer.zero_grad()

                train_loss = self._calculate_timeseries_loss(ytrain_true, list_ytrain_pred)

                train_loss.backward()
                self.optimizer.step()
                train_epoch_loss += train_loss.item() 

            # VALIDATION    
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_epoch_loss = 0
                for batch, sample in enumerate(data_loader_validate):
                    
                    xvalid = sample['image'].to(device)
                    yvalid_true = sample['mask'][:,:,:,:,0].to(device)
                    embmatrix_valid = sample['EmbMatrix'].to(device)

                    list_yvalid_pred   = self.model(xvalid, embmatrix_valid) 
  
                    valid_loss = self._calculate_timeseries_loss(yvalid_true, list_yvalid_pred)

                    val_epoch_loss += valid_loss.item()

            loss_stats['train'].append(train_epoch_loss/len(data_loader_training))
            loss_stats['val'].append(val_epoch_loss/len(data_loader_validate))

            training_duration_time = (time.time() - training_start_time)        
            print(f'Epoch {epoch+0:03}: | Time(s): {training_duration_time:.3f}| Train Loss: {train_epoch_loss/len(data_loader_training):.4f} | Val Loss: {val_epoch_loss/len(data_loader_validate):.4f}') 


            checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': best_val_loss
            }

            save_checkpoint(checkpoint, filename= os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

            if (val_epoch_loss/len(data_loader_validate)) < best_val_loss or epoch==0:
                        
                best_val_loss=(val_epoch_loss/len(data_loader_validate))
                torch.save(self.model.state_dict(), self.best_model_name)

                save_checkpoint(checkpoint, filename = self.best_checkpoint_dir)

                # early_stopping.update(False)
                print(f'=============================== Best model Saved! Val MSE: {best_val_loss:.4f}')

                status = True
            else:

                status = False

            early_stopping(status)
            if early_stopping.early_stop:
                print("Early stopping triggered at epoch:", epoch)
                torch.save(self.model.state_dict(), self.last_model_name)
                break

        save_loss_df(loss_stats, self.loss_df_name, self.loss_fig_name)

    def _calculate_timeseries_loss(self, y_true, list_y_pred):
        """
        Calculates the cumulative mean squared error loss for a list of predictions or a single prediction.

        Parameters:
        - y_true (Tensor): The ground truth values.
        - list_y_pred (list of Tensors or Tensor): List of predicted values or a single tensor.
        - weight (Tensor): The weight for weighted loss calculations.
        - loss_type (str): The type of loss ('mse', 'wmse', 'huber').

        Returns:
        - Tensor: The cumulative MSE loss for all predictions.
        """

        # Check if list_y_pred is a list or a single tensor

        if isinstance(list_y_pred, list):
            total_loss = 0.0
            for y_pred in list_y_pred:

                current_loss = mse_loss(y_pred, y_true)
                total_loss += current_loss
        else:
            # list_y_pred is a single tensor
            y_pred = list_y_pred
            total_loss = mse_loss(y_pred, y_true) #

        return total_loss

    def predict(self, model, data_loader, category: str, iter: int):

        model.load_state_dict(torch.load(self.best_model_name))
        output_files =[]

        for i in range(iter):
            with torch.no_grad():
                for sample in data_loader:
                    x = sample['image'].to(device)
                    y = sample['mask'].detach().cpu().numpy()
                    block_id = sample['block']
                    block_cultivar_id = sample['cultivar']
                    block_x_coords = sample['X']
                    block_y_coords = sample['Y']
                    embmatrix = sample['EmbMatrix'].to(device)
                
                    pred_list = self.model(x, embmatrix)

                    this_batch = {"block": block_id, 
                                        "cultivar": block_cultivar_id, 
                                        "X": block_x_coords, "Y": block_y_coords,
                                        "ytrue": y, 
                                        "ypred_w1": pred_list[0].detach().cpu().numpy(),
                                        "ypred_w2": pred_list[1].detach().cpu().numpy(),
                                        "ypred_w3": pred_list[2].detach().cpu().numpy(),
                                        "ypred_w4": pred_list[3].detach().cpu().numpy(),
                                        "ypred_w5": pred_list[4].detach().cpu().numpy(),
                                        "ypred_w6": pred_list[5].detach().cpu().numpy(),
                                        "ypred_w7": pred_list[6].detach().cpu().numpy(),
                                        "ypred_w8": pred_list[7].detach().cpu().numpy(),
                                        "ypred_w9": pred_list[8].detach().cpu().numpy(),
                                        "ypred_w10": pred_list[9].detach().cpu().numpy(),
                                        "ypred_w11": pred_list[10].detach().cpu().numpy(),
                                        "ypred_w12": pred_list[11].detach().cpu().numpy(),
                                        "ypred_w13": pred_list[12].detach().cpu().numpy(),
                                        "ypred_w14": pred_list[13].detach().cpu().numpy(),
                                        "ypred_w15": pred_list[14].detach().cpu().numpy()}
                    
                    output_files.append(this_batch)

                modified_df = self._return_modified_pred_df(output_files, None, 16)
                if category == 'train':
                    name_tr = self.train_df_name[:-4] + f'_{i}' + '.csv'
                    modified_df.to_csv(name_tr)
                    print("train inference is done!")
                elif category == 'valid':
                    name_val = self.valid_df_name[:-4] + f'_{i}' + '.csv'
                    modified_df.to_csv(name_val)
                    print("validation inference is done!")
                elif category == 'test':
                    name_te = self.test_df_name[:-4] + f'_{i}' + '.csv'
                    modified_df.to_csv(name_te)
                    print("test inference is done!")

    def _return_modified_pred_df(self, pred_npy, blocks_list, wsize = None):

        if blocks_list is None: 
            all_block_names = [dict['block'] for dict in pred_npy]#[0]
            blocks_list = list(set(item for sublist in all_block_names for item in sublist))

        OutDF = pd.DataFrame()
        out_ytrue, out_blocks, out_cultivars, out_x, out_y = [], [], [], [], []
        out_ypred_w1, out_ypred_w2, out_ypred_w3, out_ypred_w4, out_ypred_w5 = [], [], [], [], []
        out_ypred_w6, out_ypred_w7, out_ypred_w8, out_ypred_w9, out_ypred_w10 = [], [], [], [], []
        out_ypred_w11, out_ypred_w12, out_ypred_w13, out_ypred_w14, out_ypred_w15 = [], [], [], [], []

        out_ypreds = {f'ypred_w{p}': [] for p in range(1, 16, 1)}
        
        for block in blocks_list:  
            
            name_split = os.path.split(block)[-1]
            block_name = name_split.replace(name_split[7:], '')
            root_name = name_split.replace(name_split[:4], '').replace(name_split[3], '')
            block_id = root_name
            
            res = {key: configs.blocks_information[key] for key in configs.blocks_information.keys() & {block_name}}
            list_d = res.get(block_name)
            cultivar_id = list_d[1]

            for l in range(len(pred_npy)):
                tb_pred_indices = [i for i, x in enumerate(pred_npy[l]['block']) if x == block]
                if len(tb_pred_indices) !=0:   
                    for index in tb_pred_indices:

                        x0 = pred_npy[l]['X'][index]
                        y0 = pred_npy[l]['Y'][index]
                        x_vector, y_vector = self.xy_vector_generator(x0, y0, wsize)
                        out_x.append(x_vector)
                        out_y.append(y_vector)
        
                        tb_ytrue = pred_npy[l]['ytrue'][index]
                        tb_flatten_ytrue = tb_ytrue.flatten()
                        out_ytrue.append(tb_flatten_ytrue)

                        tb_ypred_w1 = pred_npy[l]['ypred_w1'][index]
                        tb_flatten_ypred_w1 = tb_ypred_w1.flatten()
                        out_ypred_w1.append(tb_flatten_ypred_w1)

                        tb_ypred_w2 = pred_npy[l]['ypred_w2'][index]
                        tb_flatten_ypred_w2 = tb_ypred_w2.flatten()
                        out_ypred_w2.append(tb_flatten_ypred_w2)

                        tb_ypred_w3 = pred_npy[l]['ypred_w3'][index]
                        tb_flatten_ypred_w3 = tb_ypred_w3.flatten()
                        out_ypred_w3.append(tb_flatten_ypred_w3)

                        tb_ypred_w4 = pred_npy[l]['ypred_w4'][index]
                        tb_flatten_ypred_w4 = tb_ypred_w4.flatten()
                        out_ypred_w4.append(tb_flatten_ypred_w4)

                        tb_ypred_w5 = pred_npy[l]['ypred_w5'][index]
                        tb_flatten_ypred_w5 = tb_ypred_w5.flatten()
                        out_ypred_w5.append(tb_flatten_ypred_w5)

                        tb_ypred_w6 = pred_npy[l]['ypred_w6'][index]
                        tb_flatten_ypred_w6 = tb_ypred_w6.flatten()
                        out_ypred_w6.append(tb_flatten_ypred_w6)

                        tb_ypred_w7 = pred_npy[l]['ypred_w7'][index]
                        tb_flatten_ypred_w7 = tb_ypred_w7.flatten()
                        out_ypred_w7.append(tb_flatten_ypred_w7)

                        tb_ypred_w8 = pred_npy[l]['ypred_w8'][index]
                        tb_flatten_ypred_w8 = tb_ypred_w8.flatten()
                        out_ypred_w8.append(tb_flatten_ypred_w8)

                        tb_ypred_w9 = pred_npy[l]['ypred_w9'][index]
                        tb_flatten_ypred_w9 = tb_ypred_w9.flatten()
                        out_ypred_w9.append(tb_flatten_ypred_w9)

                        tb_ypred_w10 = pred_npy[l]['ypred_w10'][index]
                        tb_flatten_ypred_w10 = tb_ypred_w10.flatten()
                        out_ypred_w10.append(tb_flatten_ypred_w10)

                        tb_ypred_w11 = pred_npy[l]['ypred_w11'][index]
                        tb_flatten_ypred_w11 = tb_ypred_w11.flatten()
                        out_ypred_w11.append(tb_flatten_ypred_w11)

                        tb_ypred_w12 = pred_npy[l]['ypred_w12'][index]
                        tb_flatten_ypred_w12 = tb_ypred_w12.flatten()
                        out_ypred_w12.append(tb_flatten_ypred_w12)

                        tb_ypred_w13 = pred_npy[l]['ypred_w13'][index]
                        tb_flatten_ypred_w13 = tb_ypred_w13.flatten()
                        out_ypred_w13.append(tb_flatten_ypred_w13)

                        tb_ypred_w14 = pred_npy[l]['ypred_w14'][index]
                        tb_flatten_ypred_w14 = tb_ypred_w14.flatten()
                        out_ypred_w14.append(tb_flatten_ypred_w14)

                        tb_ypred_w15 = pred_npy[l]['ypred_w15'][index]
                        tb_flatten_ypred_w15 = tb_ypred_w15.flatten()
                        out_ypred_w15.append(tb_flatten_ypred_w15)

                        tb_block_id = np.array(len(tb_flatten_ytrue)*[block_id], dtype=np.int32)
                        out_blocks.append(tb_block_id)

                        tb_cultivar_id = np.array(len(tb_flatten_ytrue)*[cultivar_id], dtype=np.int8)
                        out_cultivars.append(tb_cultivar_id)

        out_blocks = np.concatenate(out_blocks)
        out_cultivars = np.concatenate(out_cultivars)
        out_x = np.concatenate(out_x)
        out_y = np.concatenate(out_y)
        out_ytrue = np.concatenate(out_ytrue)
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
        
        OutDF['block'] = out_blocks
        OutDF['cultivar'] = out_cultivars
        OutDF['x'] = out_x
        OutDF['y'] = out_y
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
    
    def xy_vector_generator(self, x0, y0, wsize):


        x_vector, y_vector = [], []
        
        for i in range(x0, x0+wsize):
            for j in range(y0, y0+wsize):
                x_vector.append(i)
                y_vector.append(j)

        return x_vector, y_vector 
    




