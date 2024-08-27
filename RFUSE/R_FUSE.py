import os
import inspect
from scipy import io
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .network import RFUSEModel
from .losses import SpectralLoss, StructLoss
from .input_preprocessing import normalize, denormalize, input_prepro_fr


from Utils.dl_tools import open_config, generate_paths, TrainingDataset20mRR


def R_FUSE(ordered_dict):
    bands_10 = torch.clone(ordered_dict.bands_10).float()
    bands_20 = torch.clone(ordered_dict.bands_20).float()

    if bands_20.shape[1] < 6:
        return torch.zeros(bands_20.shape[0], bands_20.shape[1], bands_10.shape[2], bands_10.shape[3], device=bands_20.device, dtype=bands_20.dtype)

    config_path = os.path.join(os.path.dirname(inspect.getfile(RFUSEModel)), 'config.yaml')
    config = open_config(config_path)

    ratio = 2

    #os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda:" + config.gpu_number if torch.cuda.is_available() else "cpu")

    model_weights_path = config.model_weights_path

    net = RFUSEModel(config.number_bands_10, config.number_bands_20)

    if not config.train or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join('RFUSE', 'weights', 'R-FUSE.tar')
        net.load_state_dict(torch.load(model_weights_path))

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
            if not os.path.exists('./Stats/R-FUSE'):
                os.makedirs('./Stats/R-FUSE')
            io.savemat('./Stats/R-FUSE/Training_20m.mat', history)

    # Target Adaptive Phase

    mean = torch.mean(bands_20, dim=(2, 3), keepdim=True)
    std = torch.std(bands_20, dim=(2, 3), keepdim=True)

    bands_10_norm = normalize(bands_10)
    bands_20_norm = normalize(bands_20)

    bands_10_norm, bands_20_norm, spec_ref, struct_ref = input_prepro_fr(bands_10_norm, bands_20_norm, ratio)

    input_10 = bands_10_norm.to(device)
    input_20 = bands_20_norm.to(device)
    spec_ref = spec_ref.to(device)
    struct_ref = struct_ref.to(device)

    net, ta_history = target_adaptation(device, net, input_10, input_20, spec_ref, struct_ref, config)

    if config.ta_save_weights:
        if not os.path.exists(config.ta_save_weights_path):
            os.makedirs(config.ta_save_weights_path)
        torch.save(net.state_dict(), config.ta_save_weights_path)

    if config.ta_save_training_stats:
        if not os.path.exists('./Stats/R-FUSE'):
            os.makedirs('./Stats/R-FUSE')
        io.savemat('./Stats/R-FUSE/TA_R-Fuse.mat', ta_history)

    net.eval()
    with torch.no_grad():
        fused = net(input_10, input_20)

    fused = denormalize(fused, mean, std)

    torch.cuda.empty_cache()

    return fused.cpu().detach()


def train(device, net, train_loader, config, val_loader=None):
    criterion = nn.L1Loss(reduction='mean')
    optim = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    net = net.to(device)

    history_loss = []
    history_val_loss = []

    pbar = tqdm(range(config.epochs))

    for epoch in pbar:

        pbar.set_description('Epoch %d/%d' % (epoch + 1, config.epochs))
        running_loss = 0.0
        running_val_loss = 0.0

        net.train()

        for i, data in enumerate(train_loader):
            optim.zero_grad()

            inputs_10, inputs_20, labels = data
            inputs_10 = inputs_10.to(device)
            inputs_20 = inputs_20.to(device)
            labels = labels.to(device)

            outputs = net(inputs_10, inputs_20)

            loss = criterion(outputs, labels)

            loss.backward()
            optim.step()
            running_loss += loss.item()

        running_loss = running_loss / len(train_loader)

        if val_loader is not None:
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs_10, inputs_20, labels = data
                    inputs_10 = inputs_10.to(device)
                    inputs_20 = inputs_20.to(device)
                    labels = labels.to(device)

                    outputs = net(inputs_10, inputs_20)

                    val_loss = criterion(outputs, labels)
                    running_val_loss += val_loss.item()

            running_val_loss = running_val_loss / len(val_loader)

        history_loss.append(running_loss)
        history_val_loss.append(running_val_loss)

        pbar.set_postfix(
            {'loss': running_loss, 'val loss': running_val_loss})

    history = {'loss': history_loss, 'val_loss': history_val_loss}

    return net, history


def target_adaptation(device, net, input_10, input_20, spectral_ref, struct_ref, config):
    optim = torch.optim.Adam(net.parameters(), lr=config.ta_learning_rate)
    net = net.to(device)
    input_10 = input_10.to(device)
    input_20 = input_20.to(device)
    spectral_ref = spectral_ref.to(device)
    struct_ref = struct_ref.to(device)
    spec_criterion = SpectralLoss().to(device)
    struct_criterion = StructLoss().to(device)

    history_spec_loss = []
    history_struct_loss = []

    net.train()

    pbar = tqdm(range(config.ta_epochs))
    for _ in pbar:
        optim.zero_grad()
        outputs = net(input_10, input_20)
        spec_loss = spec_criterion(outputs, spectral_ref)
        struct_loss = struct_criterion(outputs, struct_ref)
        loss = config.lambda_1 * spec_loss + config.lambda_2 * struct_loss
        loss.backward()
        optim.step()

        history_spec_loss.append(spec_loss.item())
        history_struct_loss.append(struct_loss.item())

        pbar.set_postfix(
            {'spec loss': spec_loss.item(), 'struct loss': struct_loss.item()})

    history = {'spec_loss': history_spec_loss, 'struct_loss': history_struct_loss}

    return net, history
