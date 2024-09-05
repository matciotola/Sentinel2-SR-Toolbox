import os
from scipy import io
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import inspect

from .network import FUSEModel
from .losses import SpectralLoss, StructLoss, RegLoss
from .input_preprocessing import normalize, denormalize, upsample_protocol

from Utils.dl_tools import open_config, generate_paths, TrainingDataset20mRR, TrainingDataset60mRR


def FUSE(ordered_dict):
    if ordered_dict.ratio == 2:
        return FUSE_20(ordered_dict)
    else:
        return FUSE_60(ordered_dict)


def FUSE_20(ordered_dict):
    bands_10 = torch.clone(ordered_dict.bands_10).float()
    bands_20 = torch.clone(ordered_dict.bands_20).float()

    config_path = os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), 'config.yaml')
    config = open_config(config_path)

    #os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda:" + config.gpu_number if torch.cuda.is_available() else "cpu")

    model_weights_path = config.model_weights_path

    net = FUSEModel(config.number_bands_10, config.number_bands_20, ratio=ordered_dict.ratio)

    if not (config.train and ordered_dict.img_number == 0) or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), 'weights',
                                              ordered_dict.dataset + '_20.tar')
        if os.path.exists(model_weights_path):
            net.load_state_dict(torch.load(model_weights_path))
            print('Weights loaded from: ' + model_weights_path)

    net = net.to(device)

    if (config.train or config.resume) and ordered_dict.img_number == 0:
        if config.training_img_root == '':
            training_img_root = ordered_dict.root
        else:
            training_img_root = config.training_img_root
        train_paths = generate_paths(training_img_root, ordered_dict.dataset, 'Training', os.path.join('Reduced_Resolution', '20'))
        ds_train = TrainingDataset20mRR(train_paths, normalize)
        train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)

        if config.validation:
            val_paths = generate_paths(training_img_root, ordered_dict.dataset, 'Validation', os.path.join('Reduced_Resolution', '20'))
            ds_val = TrainingDataset20mRR(val_paths, normalize)
            val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=True)
        else:
            val_loader = None

        history = train(device, net, train_loader, config, ordered_dict, val_loader)

        if config.save_weights:
            if not os.path.exists(
                    os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), config.save_weights_path)):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), config.save_weights_path))
            torch.save(net.state_dict(),
                       os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), config.save_weights_path,
                                    ordered_dict.dataset + '_20.tar'))

        if config.save_training_stats:
            if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), 'Stats', 'FUSE')):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), 'Stats', 'FUSE'))
            io.savemat(
                os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), 'Stats', 'FUSE', 'Training_FUSE_20.mat'),
                history)

    bands_10_norm = normalize(bands_10)
    bands_20_up = upsample_protocol(bands_20, ratio)
    bands_20_up_norm = normalize(bands_20_up)

    input_10 = bands_10_norm.to(device)
    input_20 = bands_20_up_norm.to(device)
    net.eval()
    with torch.no_grad():
        fused = net(input_10, input_20)

    fused = denormalize(fused)

    torch.cuda.empty_cache()
    return fused.cpu().detach()


def FUSE_60(ordered_dict):
    bands_10 = torch.clone(ordered_dict.bands_10).float()
    bands_60 = torch.clone(ordered_dict.bands_60).float()

    config_path = os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), 'config.yaml')
    config = open_config(config_path)

    #os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda:" + config.gpu_number if torch.cuda.is_available() else "cpu")

    model_weights_path = config.model_weights_path

    net = FUSEModel(config.number_bands_10, config.number_bands_60, ratio=ordered_dict.ratio)

    if not (config.train and ordered_dict.img_number == 0) or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), 'weights',
                                              ordered_dict.dataset + '_60.tar')
        if os.path.exists(model_weights_path):
            net.load_state_dict(torch.load(model_weights_path))
            print('Weights loaded from: ' + model_weights_path)

    net = net.to(device)

    if (config.train or config.resume) and ordered_dict.img_number == 0:
        if config.training_img_root == '':
            training_img_root = ordered_dict.root
        else:
            training_img_root = config.training_img_root
        train_paths = generate_paths(training_img_root, ordered_dict.dataset, 'Training', os.path.join('Reduced_Resolution', '60'))
        ds_train = TrainingDataset60mRR(train_paths, normalize)
        train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)

        if config.validation:
            val_paths = generate_paths(training_img_root, ordered_dict.dataset, 'Validation', os.path.join('Reduced_Resolution', '60'))
            ds_val = TrainingDataset60mRR(val_paths, normalize)
            val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=True)
        else:
            val_loader = None

        history = train(device, net, train_loader, config, ordered_dict, val_loader)

        if config.save_weights:
            if not os.path.exists(
                    os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), config.save_weights_path)):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), config.save_weights_path))
            torch.save(net.state_dict(),
                       os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), config.save_weights_path,
                                    ordered_dict.dataset + '_60.tar'))

        if config.save_training_stats:
            if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), 'Stats', 'FUSE')):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), 'Stats', 'FUSE'))
            io.savemat(
                os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), 'Stats', 'FUSE', 'Training_FUSE_20.mat'),
                history)

    bands_10_norm = normalize(bands_10)
    bands_60_up = upsample_protocol(bands_60, ordered_dict.ratio)
    bands_60_up_norm = normalize(bands_60_up)

    input_10 = bands_10_norm.to(device)
    input_60 = bands_60_up_norm.to(device)
    net.eval()
    with torch.no_grad():
        fused = net(input_10, input_60)

    fused = denormalize(fused)

    torch.cuda.empty_cache()
    return fused.cpu().detach()


def train(device, net, train_loader, config, ordered_dict, val_loader=None):
    criterion_spec = SpectralLoss().to(device)
    criterion_struct = StructLoss().to(device)
    criterion_reg = RegLoss().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2))

    history_loss_spec = []
    history_loss_struct = []
    history_loss_reg = []

    history_val_loss_spec = []
    history_val_loss_struct = []
    history_val_loss_reg = []

    pbar = tqdm(range(config.epochs))

    for epoch in pbar:

        pbar.set_description('Epoch %d/%d' % (epoch + 1, config.epochs))
        running_loss_spec = 0.0
        running_loss_struct = 0.0
        running_loss_reg = 0.0

        running_val_loss_spec = 0.0
        running_val_loss_struct = 0.0
        running_val_loss_reg = 0.0

        net.train()

        for i, data in enumerate(train_loader):
            optim.zero_grad()

            if len(data) == 3:
                inputs_10, inputs_20, labels = data
                inputs_lr = inputs_20
            else:
                inputs_10, _, inputs_60, labels = data
                inputs_lr = inputs_60

            inputs_10 = inputs_10.float().to(device)
            inputs_lr = upsample_protocol(inputs_lr, ordered_dict.ratio).float().to(device)
            labels = labels.float().to(device)

            outputs = net(inputs_10, inputs_lr)

            loss_spec = criterion_spec(outputs, labels)
            loss_struct = criterion_struct(outputs, labels)
            loss_reg = criterion_reg(outputs)

            loss = config.lambda_1 * loss_spec + config.lambda_2 * loss_struct + config.lambda_3 * loss_reg

            loss.backward()
            optim.step()
            running_loss_spec += loss_spec.item()
            running_loss_struct += loss_struct.item()
            running_loss_reg += loss_reg.item()

        running_loss_spec = running_loss_spec / len(train_loader)
        running_loss_struct = running_loss_struct / len(train_loader)
        running_loss_reg = running_loss_reg / len(train_loader)

        if val_loader is not None:
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    if len(data) == 3:
                        inputs_10, inputs_20, labels = data
                        inputs_lr = inputs_20
                    else:
                        inputs_10, _, inputs_60, labels = data
                        inputs_lr = inputs_60

                    inputs_10 = inputs_10.float().to(device)
                    inputs_lr = upsample_protocol(inputs_lr, ordered_dict.ratio).float().to(device)
                    labels = labels.float().to(device)

                    outputs = net(inputs_10, inputs_lr)

                    val_loss_spec = criterion_spec(outputs, labels)
                    val_loss_struct = criterion_struct(outputs, labels)
                    val_loss_reg = criterion_reg(outputs)

                    running_val_loss_spec += val_loss_spec.item()
                    running_val_loss_struct += val_loss_struct.item()
                    running_val_loss_reg += val_loss_reg.item()

            running_val_loss_spec = running_val_loss_spec / len(val_loader)
            running_val_loss_struct = running_val_loss_struct / len(val_loader)
            running_val_loss_reg = running_val_loss_reg / len(val_loader)

        history_loss_spec.append(running_loss_spec)
        history_loss_struct.append(running_loss_struct)
        history_loss_reg.append(running_loss_reg)
        history_val_loss_spec.append(running_val_loss_spec)
        history_val_loss_struct.append(running_val_loss_struct)
        history_val_loss_reg.append(running_val_loss_reg)

        pbar.set_postfix(
            {'spec loss': running_loss_spec, 'struct loss': running_loss_struct, 'reg loss': running_loss_reg,
             'val spec loss': running_val_loss_spec, 'val struct loss': running_val_loss_struct,
             'val reg loss': running_val_loss_reg})

    history = {'spec_loss': history_loss_spec, 'struct_loss': history_loss_struct, 'reg_loss': history_loss_reg,
               'val_spec_loss': history_val_loss_spec, 'val_struct_loss': history_val_loss_struct,
               'val_reg_loss': history_val_loss_reg}

    return history
