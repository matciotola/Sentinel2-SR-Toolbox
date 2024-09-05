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
    bands_10 = torch.clone(ordered_dict.bands_10).float()
    bands_20 = torch.clone(ordered_dict.bands_20).float()

    if bands_20.shape[1] < 6:
        return torch.zeros(bands_20.shape[0], bands_20.shape[1], bands_10.shape[2], bands_10.shape[3], device=bands_20.device, dtype=bands_20.dtype)

    config_path = os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), 'config.yaml')
    config = open_config(config_path)

    ratio = 2

    #os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda:" + config.gpu_number if torch.cuda.is_available() else "cpu")

    model_weights_path = config.model_weights_path

    net = FUSEModel(config.number_bands_10, config.number_bands_20)

    if not (config.train and ordered_dict.img_number == 0) or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join(os.path.dirname(inspect.getfile(FUSEModel)), 'weights',
                                              ordered_dict.dataset + '.tar')
        if os.path.exists(model_weights_path):
            net.load_state_dict(torch.load(model_weights_path))
            print('Weights loaded from: ' + model_weights_path)

    net = net.to(device)

    if config.train:
        train_paths = generate_paths(config.training_img_root, 'Reduced_Resolution', 'Training',  '20')
        ds_train = TrainingDataset20mRR(train_paths, normalize)
        train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)

        if len(config.validation_img_names) != 0:
            val_paths = generate_paths(config.training_img_root, 'Reduced_Resolution', 'Validation',  '20')
            ds_val = TrainingDataset20mRR(val_paths, normalize)
            val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=True)
        else:
            val_loader = None

        net, history = train(device, net, train_loader, config, val_loader)

        if config.save_weights:
            if not os.path.exists(config.save_weights_path):
                os.makedirs(config.save_weights_path)
            torch.save(net.state_dict(), config.save_weights_path)

        if config.save_training_stats:
            if not os.path.exists('./Stats/FUSE'):
                os.makedirs('./Stats/FUSE')
            io.savemat('./Stats/FUSE/Training_FUSE.mat', history)

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


def train(device, net, train_loader, config, val_loader=None):
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

            inputs_10, inputs_20, labels = data
            inputs_10 = inputs_10.to(device)
            inputs_20 = inputs_20.to(device)
            inputs_20 = upsample_protocol(inputs_20, 2)
            labels = labels.to(device)

            outputs = net(inputs_10, inputs_20)

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
                    inputs_10, inputs_20, labels = data
                    inputs_10 = inputs_10.to(device)
                    inputs_20 = inputs_20.to(device)
                    labels = labels.to(device)

                    outputs = net(inputs_10, inputs_20)

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

    return net, history
