import os
from scipy import io
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from network import FUSEModel
    from losses import SpectralLoss, StructLoss, RegLoss
    from input_preprocessing import normalize, denormalize, input_prepro_20, upsample_protocol, get_patches
except:
    from FUSE.network import FUSEModel
    from FUSE.losses import SpectralLoss, StructLoss, RegLoss
    from FUSE.input_preprocessing import normalize, denormalize, input_prepro_20, upsample_protocol, get_patches

from Utils.dl_tools import open_config, generate_paths, TrainingDataset20m




def FUSE(ordered_dict):

    bands_high = torch.clone(ordered_dict.bands_high)
    bands_low_lr = torch.clone(ordered_dict.bands_low_lr)

    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        config_path = os.path.join('FUSE', 'config.yaml')

    config = open_config(config_path)
    ratio = 2

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_weights_path = config.model_weights_path

    net = FUSEModel((config.number_bands_10, config.number_bands_20))

    if not config.train or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join('FUSE', 'weights', 'FUSE.tar')
        net.load_state_dict(torch.load(model_weights_path))

    net = net.to(device)

    if config.train:
        train_paths_10, train_paths_20, _ = generate_paths(config.training_img_root, config.training_img_names)
        ds_train = TrainingDataset20m(train_paths_10, train_paths_20, normalize, input_prepro_20, get_patches, ratio, config.training_patch_size_20)
        train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)

        if len(config.validation_img_names) != 0:
            val_paths_10, val_paths_20, _ = generate_paths(config.validation_img_root, config.validation_img_names)
            ds_val = TrainingDataset20m(val_paths_10, val_paths_20, normalize, input_prepro_20, get_patches, ratio, config.training_patch_size_20)
            val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=True)
        else:
            val_loader = None

        net, history = train(net, train_loader, val_loader)

        if config.save_weights:
            torch.save(net.state_dict(), config.save_weights_path)

        if config.save_training_stats:
            if not os.path.exists('./Stats/FUSE'):
                os.makedirs('./Stats/FUSE')
            io.savemat('./Stats/DSen2/Training_20m.mat', history)


    bands_high_norm = normalize(bands_high)
    bands_low = upsample_protocol(bands_low_lr, bands_high_norm.shape)
    bands_low_norm = normalize(bands_low)


    input_10 = bands_high_norm.to(device)
    input_20 = bands_low_norm.to(device)
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

        pbar.set_postfix({'spec loss': running_loss_spec, 'struct loss': running_loss_struct, 'reg loss': running_loss_reg, 'val spec loss': running_val_loss_spec, 'val struct loss': running_val_loss_struct, 'val reg loss': running_val_loss_reg})

    history = {'spec_loss': history_loss_spec, 'struct_loss': history_loss_struct, 'reg_loss': history_loss_reg, 'val_spec_loss': history_val_loss_spec, 'val_struct_loss': history_val_loss_struct, 'val_reg_loss': history_val_loss_reg}

    return net, history