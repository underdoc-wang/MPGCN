import os
import argparse
import Data_Container_OD, Model_Trainer



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run OD Prediction.')

    # command line arguments
    parser.add_argument('-GPU', '--GPU', type=str, help='Specify GPU usage', default='cuda:2')      # 'cpu' for no GPU used
    parser.add_argument('-in', '--input_dir', type=str, default='../data')
    parser.add_argument('-out', '--output_dir', type=str, default='./output')
    parser.add_argument('-model', '--model', type=str, help='Specify model', choices=['MPGCN'], default='MPGCN')
    parser.add_argument('-t', '--time_slice', type=int, help='Temporal granularity', default=24)        # daily
    parser.add_argument('-obs', '--obs_len', type=int, help='Length of observation sequence', default=7)
    parser.add_argument('-pred', '--pred_len', type=int, help='Length of prediction sequence', default=7)
    parser.add_argument('-norm', '--norm', type=str, choices=['none', 'minmax', 'std'], default='none')
    parser.add_argument('-split', '--split_ratio', type=int, nargs='+',
                        help='Relative data split ratio in train : validate : test'
                             ' Example: -split 5 1 2',
                        default=[6.4, 1.6, 2])
    parser.add_argument('-batch', '--batch_size', type=int, default=4)
    parser.add_argument('-hidden', '--hidden_dim', type=int, default=32)
    parser.add_argument('-kernel', '--kernel_type', type=str,
                        choices=['chebyshev', 'localpool', 'random_walk_diffusion', 'dual_random_walk_diffusion'],
                        default='random_walk_diffusion')    # GCN kernel type
    parser.add_argument('-K', '--cheby_order', type=int, default=2)    # GCN chebyshev order

    parser.add_argument('-nn', '--nn_layers', type=int, default=2)        # layers
    parser.add_argument('-loss', '--loss', type=str, help='Specify loss function',
                        choices=['MSE', 'MAE', 'Huber'], default='MSE')
    parser.add_argument('-optim', '--optimizer', type=str, help='Specify optimizer', default='Adam')
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-4)
    parser.add_argument('-dr', '--decay_rate', type=float, default=0)    # weight decay: L2regularization
    parser.add_argument('-epoch', '--num_epochs', type=int, default=200)
    parser.add_argument('-mode', '--mode', type=str, choices=['train', 'test'], default='train')

    params = parser.parse_args().__dict__       # save in dict

    # paths
    os.makedirs(params['output_dir'], exist_ok=True)

    if params['mode'] == 'train':
        params['pred_len'] = 1      # train single-step model

    # load data
    data_input = Data_Container_OD.DataInput(params=params)
    data = data_input.load_data()
    params['N'] = data['OD'].shape[1]

    # get data loader
    data_generator = Data_Container_OD.DataGenerator(obs_len=params['obs_len'],
                                                        pred_len=params['pred_len'],
                                                        data_split_ratio=params['split_ratio'])
    data_loader = data_generator.get_data_loader(data=data,
                                                 params=params)

    # get model
    trainer = Model_Trainer.ModelTrainer(params=params,
                                         data=data,
                                         data_container=data_input)

    if params['mode'] == 'train':
        trainer.train(data_loader=data_loader, modes=['train', 'validate'])
    else:   # test
        trainer.test(data_loader=data_loader, modes=['train', 'test'])


