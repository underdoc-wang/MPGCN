import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import distance
import scipy.sparse as ss



class DataInput(object):
    def __init__(self, params:dict):
        self.params = params

    def load_data(self):
        prov_day_data = ss.load_npz(self.params['input_dir'] + '/od_day20180101_20210228.npz')
        prov_day_data_dense = np.array(prov_day_data.todense()).reshape((-1, 47, 47))
        OD_DAYS = [date.strftime('%Y-%m-%d') for date in pd.date_range(start='2020-01-01', end='2021-02-28', freq='1D')]    # 425 days
        data = prov_day_data_dense[-len(OD_DAYS):, :, :, np.newaxis]
        OD_data = np.log(data + 1.0)        # log transformation
        print(OD_data.shape)

        if self.params['norm']=='none':
            pass
        elif self.params['norm']=='minmax':
            OD_data = self.minmax_normalize(OD_data)
        elif self.params['norm']=='std':
            OD_data = self.std_normalize(OD_data)
        else:
            raise ValueError

        # return a dict
        dataset = dict()
        dataset['OD'] = OD_data
        dataset['adj'] = np.load(self.params['input_dir'] + '/adjacency_matrix.npy')
        dataset['O_dyn_G'], dataset['D_dyn_G'] = self.construct_dyn_G(data)    # use unnormalized OD

        return dataset

    def construct_dyn_G(self, OD_data:np.array, perceived_period:int=7):        # construct dynamic graphs based on OD history
        train_len = int(OD_data.shape[0] * self.params['split_ratio'][0] / sum(self.params['split_ratio']))
        num_periods_in_history = train_len // perceived_period      # dump the remainder
        OD_history = OD_data[:num_periods_in_history * perceived_period, :,:,:]

        O_dyn_G, D_dyn_G = [], []
        for t in range(perceived_period):
            OD_t_avg = np.mean(OD_history[t::perceived_period,:,:,:], axis=0).squeeze(axis=-1)
            O, D = OD_t_avg.shape

            O_G_t = np.zeros((O, O))    # initialize O graph at t
            for i in range(O):
                for j in range(O):
                    O_G_t[i, j] = distance.cosine(OD_t_avg[i,:], OD_t_avg[j,:])     # eq (6)
            D_G_t = np.zeros((D, D))    # initialize D graph at t
            for i in range(D):
                for j in range(D):
                    D_G_t[i, j] = distance.cosine(OD_t_avg[:,i], OD_t_avg[j,:])     # eq (7)
            O_dyn_G.append(O_G_t), D_dyn_G.append(D_G_t)

        return np.stack(O_dyn_G, axis=-1), np.stack(D_dyn_G, axis=-1)

    def minmax_normalize(self, x:np.array):     # normalize to [0, 1]
        self._max, self._min = x.max(), x.min()
        print('min:', self._min, 'max:', self._max)
        x = (x - self._min) / (self._max - self._min)
        return x

    def minmax_denormalize(self, x:np.array):
        x = (self._max - self._min) * x + self._min
        return x

    def std_normalize(self, x:np.array):        # normalize to N(0, 1)
        self._mean, self._std = x.mean(), x.std()
        print('mean:', round(self._mean, 4), 'std:', round(self._std, 4))
        x = (x - self._mean)/self._std
        return x

    def std_denormalize(self, x:np.array):
        x = x * self._std + self._mean
        return x



class ODDataset(Dataset):
    def __init__(self, inputs:dict, output:torch.Tensor, mode:str, mode_len:dict, obs_len:int):
        self.mode = mode
        self.mode_len = mode_len
        self.inputs, self.output = self.prepare_xy(inputs, output)
        self.obs_len = obs_len

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item:int):        # item: time index in current mode
        O_G_t, D_G_t = self.timestamp_query(self.inputs['O_dyn_G'], self.inputs['D_dyn_G'], item)
        return self.inputs['x_seq'][item], self.output[item], O_G_t, D_G_t      # dynamic graph shape: (batch, N, N)

    def timestamp_query(self, O_dyn_G:torch.Tensor, D_dyn_G:torch.Tensor, t:int, perceived_period:int=7):    # for dynamic graph at t
        # get y's timestamp relative to initial timestamp of the dataset
        if self.mode == 'train':
            timestamp = self.obs_len + t
        elif self.mode == 'validate':
            timestamp = self.obs_len + self.mode_len['train'] + t
        else:       # test
            timestamp = self.obs_len + self.mode_len['train'] + self.mode_len['validate'] + t

        key = timestamp % perceived_period
        O_G_t, D_G_t = O_dyn_G[:,:,key], D_dyn_G[:,:,key]
        return O_G_t, D_G_t

    def prepare_xy(self, inputs:dict, output:torch.Tensor):
        if self.mode == 'train':
            start_idx = 0
        elif self.mode == 'validate':
            start_idx = self.mode_len['train']
        else:       # test
            start_idx = self.mode_len['train']+self.mode_len['validate']

        x = dict()
        x['x_seq'] = inputs['x_seq'][start_idx : (start_idx + self.mode_len[self.mode])]
        x['O_dyn_G'], x['D_dyn_G'] = inputs['O_dyn_G'], inputs['D_dyn_G']
        y = output[start_idx : start_idx + self.mode_len[self.mode]]
        return x, y



class DataGenerator(object):
    def __init__(self, obs_len:int, pred_len, data_split_ratio:tuple):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_split_ratio = data_split_ratio

    def split2len(self, data_len:int):
        mode_len = dict()
        mode_len['validate'] = int(self.data_split_ratio[1]/sum(self.data_split_ratio) * data_len)
        mode_len['test'] = int(self.data_split_ratio[2]/sum(self.data_split_ratio) * data_len)
        mode_len['train'] = data_len - mode_len['validate'] - mode_len['test']
        return mode_len

    def get_data_loader(self, data:dict, params:dict):
        x_seq, y_seq = self.get_feats(data['OD'])

        feat_dict = dict()
        feat_dict['x_seq'] = torch.from_numpy(np.asarray(x_seq)).float().to(params['GPU'])
        feat_dict['O_dyn_G'], feat_dict['D_dyn_G'] = torch.from_numpy(data['O_dyn_G']).float(), torch.from_numpy(data['D_dyn_G']).float()
        y_seq = torch.from_numpy(np.asarray(y_seq)).float().to(params['GPU'])

        mode_len = self.split2len(data_len=y_seq.shape[0])

        data_loader = dict()        # data_loader for [train, validate, test]
        for mode in ['train', 'validate', 'test']:
            dataset = ODDataset(inputs=feat_dict, output=y_seq,
                                mode=mode, mode_len=mode_len, obs_len=self.obs_len)
            data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=False)
        # data loading default: single-processing | for multi-processing: num_workers=pos_int or pin_memory=True (GPU)
        # data_loader multi-processing
        return data_loader

    def get_feats(self, data:np.array):
        x, y = [], []
        for i in range(self.obs_len, data.shape[0]-self.pred_len):
            x.append(data[i-self.obs_len : i])
            y.append(data[i : i+self.pred_len])
        return x, y


