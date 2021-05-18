import numpy as np
from datetime import datetime
from torch import nn, optim
import torch
import GCN, MPGCN
import Metrics



class ModelTrainer(object):
    def __init__(self, params:dict, data:dict, data_container):
        self.params = params
        self.data_container = data_container
        self.get_static_graph(graph=data['adj'])    # initialize static graphs and K values
        self.model = self.get_model().to(params['GPU'])
        self.criterion = self.get_loss()
        self.optimizer = self.get_optimizer()

    def get_static_graph(self, graph:np.array):
        self.K = self.get_support_K(self.params['kernel_type'], self.params['cheby_order'])
        self.G = self.preprocess_adj(graph, self.params['kernel_type'], self.params['cheby_order'])
        return

    @staticmethod
    def get_support_K(kernel_type, cheby_order):
        if kernel_type == 'localpool':
            assert cheby_order == 1
            K = 1
        elif (kernel_type=='chebyshev')|(kernel_type=='random_walk_diffusion'):
            K = cheby_order + 1
        elif kernel_type == 'dual_random_walk_diffusion':
            K = cheby_order*2 + 1
        else:
            raise ValueError('Invalid kernel_type. Must be one of '
                             '[chebyshev, localpool, random_walk_diffusion, dual_random_walk_diffusion].')
        return K

    def preprocess_adj(self, adj_mtx:np.array, kernel_type, cheby_order):
        self.adj_preprocessor = GCN.Adj_Processor(kernel_type, cheby_order)
        b_adj = torch.from_numpy(adj_mtx).float().unsqueeze(dim=0)      # batch_size=1
        adj = self.adj_preprocessor.process(b_adj)
        return adj.squeeze(dim=0).to(self.params['GPU'])       # G: (support_K, N, N)


    def get_model(self):
        if self.params['model'] == 'MPGCN':
            model = MPGCN.MPGCN(M=2,        # 2 branches: one for adj; the other for dynamic O/G cosine correlation graph
                                K=self.K,
                                input_dim=1,
                                lstm_hidden_dim=self.params['hidden_dim'],
                                lstm_num_layers=1,
                                gcn_hidden_dim=self.params['hidden_dim'],
                                gcn_num_layers=3,
                                num_nodes=self.params['N'],
                                user_bias=True,
                                activation=nn.ReLU)
        else:
            raise NotImplementedError('Invalid model name.')
        return model

    def get_loss(self):
        if self.params['loss'] == 'MSE':
            criterion = nn.MSELoss(reduction='mean')
        elif self.params['loss'] == 'MAE':
            criterion = nn.L1Loss(reduction='mean')
        elif self.params['loss'] == 'Huber':
            criterion = nn.SmoothL1Loss(reduction='mean')
        else:
            raise NotImplementedError('Invalid loss function.')
        return criterion

    def get_optimizer(self):
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(params=self.model.parameters(),
                                   lr=self.params['learn_rate'],
                                   weight_decay=self.params['decay_rate'])
        else:
            raise NotImplementedError('Invalid optimizer name.')
        return optimizer


    def preprocess_dynamic_graph(self, dyn_G:torch.Tensor):
        # reuse adj_preprocessor initialized in preprocessing static graphs, otherwise needed to initiate one each batch
        return self.adj_preprocessor.process(dyn_G).to(self.params['GPU'])         # (batch, K, N, N)


    def train(self, data_loader:dict, modes:list, early_stop_patience=10):
        checkpoint = {'epoch': 0, 'state_dict': self.model.state_dict()}
        val_loss = np.inf
        patience_count = early_stop_patience

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model training begins:')
        for epoch in range(1, 1 + self.params['num_epochs']):
            running_loss = {mode: 0.0 for mode in modes}
            for mode in modes:
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                step = 0
                for x_seq, y_true, O_dyn_G, D_dyn_G in data_loader[mode]:
                    with torch.set_grad_enabled(mode=(mode=='train')):
                        if self.params['model'] == 'MPGCN':
                            dyn_OD_G = (self.preprocess_dynamic_graph(O_dyn_G), self.preprocess_dynamic_graph(D_dyn_G))
                            y_pred = self.model(x_seq=x_seq, G_list=[self.G, dyn_OD_G])
                        else:
                            raise NotImplementedError('Invalid model name.')

                        loss = self.criterion(y_pred, y_true)
                        if mode == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                    running_loss[mode] += loss * y_true.shape[0]    # loss reduction='mean': batchwise average
                    step += y_true.shape[0]
                    torch.cuda.empty_cache()

                # epoch end: evaluate on validation set for early stopping
                if mode == 'validate':
                    epoch_val_loss = running_loss[mode]/step
                    if epoch_val_loss <= val_loss:
                        print(f'Epoch {epoch}, validation loss drops from {val_loss:.5} to {epoch_val_loss:.5}. '
                              f'Update model checkpoint..')
                        val_loss = epoch_val_loss
                        checkpoint.update(epoch=epoch, state_dict=self.model.state_dict())
                        torch.save(checkpoint, self.params['output_dir']+f'/{self.params["model"]}_od.pkl')
                        patience_count = early_stop_patience
                    else:
                        print(f'Epoch {epoch}, validation loss does not improve from {val_loss:.5}.')
                        patience_count -= 1
                        if patience_count == 0:
                            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                            print(f'    Early stopping at epoch {epoch}. {self.params["model"]} model training ends.')
                            return

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model training ends.')
        torch.save(checkpoint, self.params['output_dir']+f'/{self.params["model"]}_od.pkl')
        return


    def test(self, data_loader:dict, modes:list):
        trained_checkpoint = torch.load(self.params['output_dir']+f'/{self.params["model"]}_od.pkl')
        self.model.load_state_dict(trained_checkpoint['state_dict'])
        self.model.eval()

        for mode in modes:
            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            print(f'     {self.params["model"]} model testing on {mode} data begins:')
            forecast, ground_truth = [], []
            for x_seq, y_true, O_dyn_G, D_dyn_G in data_loader[mode]:
                if self.params['model'] == 'MPGCN':
                    dyn_OD_G = (self.preprocess_dynamic_graph(O_dyn_G), self.preprocess_dynamic_graph(D_dyn_G))
                    y_pred = []
                    cur_x_seq = x_seq
                    with torch.no_grad():
                        for horizon in range(self.params['pred_len']):      # extended trained one-step model for multi-step prediction
                            step_y_pred = self.model(x_seq=cur_x_seq, G_list=[self.G, dyn_OD_G])
                            cur_x_seq = torch.cat([cur_x_seq[:,1:,:,:,:], step_y_pred], dim=1)
                            y_pred.append(step_y_pred)
                    y_pred = torch.cat(y_pred, dim=1)

                else:
                    raise NotImplementedError('Invalid model name.')

                forecast.append(y_pred.cpu().detach().numpy())
                ground_truth.append(y_true.cpu().detach().numpy())

            forecast = np.concatenate(forecast, axis=0)
            ground_truth = np.concatenate(ground_truth, axis=0)
            # denormalize
            #forecast = self.data_container.minmax_denormalize(forecast)
            #ground_truth = self.data_container.minmax_denormalize(ground_truth)
            # evaluate on metrics
            MSE, RMSE, MAE, MAPE = Metrics.evaluate(forecast, ground_truth)
            f = open(self.params['output_dir'] + '/' + self.params['model'] + '_prediction_scores.txt', 'a')
            f.write("%s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (mode, MSE, RMSE, MAE, MAPE))
            f.close()

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model testing ends.')
        return


