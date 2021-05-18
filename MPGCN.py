import torch
from torch import nn



class BDGCN(nn.Module):        # 2DGCN: handling both static and dynamic graph input
    def __init__(self, K:int, input_dim:int, hidden_dim:int, use_bias=True, activation=None):
        super(BDGCN, self).__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.activation = activation() if activation is not None else None
        self.init_params()

    def init_params(self, b_init=0.0):
        self.W = nn.Parameter(torch.empty(self.input_dim*(self.K**2), self.hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W)
        if self.use_bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)
        return

    def forward(self, X:torch.Tensor, G:torch.Tensor or tuple):
        feat_set = list()
        if type(G) == torch.Tensor:         # static graph input: (K, N, N)
            assert self.K == G.shape[-3]
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = torch.einsum('bncl,nm->bmcl', X, G[o, :, :])
                    mode_2_prod = torch.einsum('bmcl,cd->bmdl', mode_1_prod, G[d, :, :])
                    feat_set.append(mode_2_prod)

        elif type(G) == tuple:              # dynamic graph input: ((batch, K, N, N), (batch, K, N, N))
            assert (len(G) == 2) & (self.K == G[0].shape[-3] == G[1].shape[-3])
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = torch.einsum('bncl,bnm->bmcl', X, G[0][:, o, :, :])
                    mode_2_prod = torch.einsum('bmcl,bcd->bmdl', mode_1_prod, G[1][:, d, :, :])
                    feat_set.append(mode_2_prod)
        else:
            raise NotImplementedError

        _2D_feat = torch.cat(feat_set, dim=-1)
        mode_3_prod = torch.einsum('bmdk,kh->bmdh', _2D_feat, self.W)

        if self.use_bias:
            mode_3_prod += self.b
        H = self.activation(mode_3_prod) if self.activation is not None else mode_3_prod
        return H



class MPGCN(nn.Module):
    def __init__(self, M:int, K:int, input_dim:int, lstm_hidden_dim:int, lstm_num_layers:int, gcn_hidden_dim:int, gcn_num_layers:int,
                 num_nodes:int, user_bias:bool, activation=None):
        super(MPGCN, self).__init__()
        self.M = M      # input graphs
        self.K = K      # chebyshev order
        self.num_nodes = num_nodes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.gcn_num_layers = gcn_num_layers

        # initiate a branch of (LSTM, 2DGCN, FC) for each graph input
        self.branch_models = nn.ModuleList()
        for m in range(self.M):
            branch = nn.ModuleDict()
            branch['temporal'] = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_num_layers, batch_first=True)
            branch['spatial'] = nn.ModuleList()
            for n in range(gcn_num_layers):
                cur_input_dim = lstm_hidden_dim if n == 0 else gcn_hidden_dim
                branch['spatial'].append(BDGCN(K=K, input_dim=cur_input_dim, hidden_dim=gcn_hidden_dim, use_bias=user_bias, activation=activation))
            branch['fc'] = nn.Sequential(
                nn.Linear(in_features=gcn_hidden_dim, out_features=input_dim, bias=True),
                nn.ReLU())
            self.branch_models.append(branch)


    def init_hidden_list(self, batch_size:int):     # for LSTM initialization
        hidden_list = list()
        for m in range(self.M):
            weight = next(self.parameters()).data
            hidden = (weight.new_zeros(self.lstm_num_layers, batch_size * (self.num_nodes**2), self.lstm_hidden_dim),
                      weight.new_zeros(self.lstm_num_layers, batch_size * (self.num_nodes**2), self.lstm_hidden_dim))
            hidden_list.append(hidden)
        return hidden_list

    def forward(self, x_seq: torch.Tensor, G_list:list):
        '''
        :param x_seq: (batch, seq, O, D, 1)
        :param G_list: static graph (K, N, N); dynamic OD graph tuple ((batch, K, N, N), (batch, K, N, N))
        :return:
        '''
        assert (len(x_seq.shape) == 5)&(self.num_nodes == x_seq.shape[2] == x_seq.shape[3])
        assert len(G_list) == self.M
        batch_size, seq_len, _, _, i = x_seq.shape
        hidden_list = self.init_hidden_list(batch_size)

        lstm_in = x_seq.permute(0, 2, 3, 1, 4).reshape(batch_size*(self.num_nodes**2), seq_len, i)
        branch_out = list()
        for m in range(self.M):
            lstm_out, hidden_list[m] = self.branch_models[m]['temporal'](lstm_in, hidden_list[m])
            gcn_in = lstm_out[:,-1,:].reshape(batch_size, self.num_nodes, self.num_nodes, self.lstm_hidden_dim)
            for n in range(self.gcn_num_layers):
                gcn_in = self.branch_models[m]['spatial'][n](gcn_in, G_list[m])
            fc_out = self.branch_models[m]['fc'](gcn_in)
            branch_out.append(fc_out)
        # ensemble
        ensemble_out = torch.mean(torch.stack(branch_out, dim=-1), dim=-1)

        return ensemble_out.unsqueeze(dim=1)        # match dim for single-step pred

