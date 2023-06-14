import os
from scipy import io
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

try:
    from network import DSen2Model
    from input_preprocessing import normalize, denormalize, input_prepro_20, input_prepro_60, get_test_patches_20, \
        get_test_patches_60, recompose_images, upsample_protocol
except:
    from DSen2.network import DSen2Model
    from DSen2.input_preprocessing import normalize, denormalize, input_prepro_20, input_prepro_60, get_test_patches_20, \
        get_test_patches_60, recompose_images, upsample_protocol

from Utils.dl_tools import open_config, generate_paths, TrainingDataset20m, TrainingDataset60m

from FUSE.input_preprocessing import get_patches # TO DO LANARAS?


def DSen2(ordered_dict):

    if ordered_dict.bands_intermediate == None:
        return DSen2_20(ordered_dict)
    else:
        return DSen2_60(ordered_dict)

def DSen2_20(ordered_dict):
    bands_high = torch.clone(ordered_dict.bands_high)
    bands_low_lr = torch.clone(ordered_dict.bands_low_lr)

    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        config_path = os.path.join('DSen2', 'config.yaml')

    config = open_config(config_path)
    ratio = 2

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_weights_path = config.model_weights_path

    net = DSen2Model((config.number_bands_10, config.number_bands_20))

    if not config.train or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join('DSen2', 'weights', 'DSen2_20m.tar')
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

        net, history = train(device, net, train_loader, config, val_loader)

        if config.save_weights:
            if not os.path.exists(config.save_weights_path):
                os.makedirs(config.save_weights_path)
            torch.save(net.state_dict(), os.path.join(config.save_weights_path, 'DSen2_20m.tar'))

        if config.save_training_stats:
            if not os.path.exists('./Stats/DSen2'):
                os.makedirs('./Stats/DSen2')
            io.savemat('./Stats/DSen2/Training_20m.mat', history)


    bands_high_norm = normalize(bands_high)
    bands_low_lr_norm = normalize(bands_low_lr)

    patches_10, patches_20 = get_test_patches_20(bands_high_norm, bands_low_lr_norm, patchSize=config.test_patch_size_20, border=config.border_20)

    output = []

    with torch.no_grad():
        for i in range(len(patches_10)):
            input_10 = patches_10[i].unsqueeze(0).to(device)
            input_20 = patches_20[i].unsqueeze(0).to(device)
            output.append(net(input_10, input_20))

    output = torch.cat(output, dim=0)
    fused = recompose_images(output, config.border_20, bands_high.shape)
    fused = denormalize(fused)


    torch.cuda.empty_cache()
    return fused.cpu().detach()


def DSen2_60(ordered_dict):

    bands_high = torch.clone(ordered_dict.bands_high)
    bands_intermediate_lr = torch.clone(ordered_dict.bands_intermediate)
    bands_low_lr = torch.clone(ordered_dict.bands_low_lr)

    if bands_low_lr.shape[1] == 3:
        bands_low_lr = bands_low_lr[:, :-1, :, :]

    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        config_path = os.path.join('DSen2', 'config.yaml')
    config = open_config(config_path)
    ratio = 2

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_weights_path = config.model_weights_path

    net = DSen2Model((config.number_bands_10, config.number_bands_20, config.number_bands_60))

    if not config.train or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join('DSen2', 'weights', 'DSen2_60m.tar')
        net.load_state_dict(torch.load(model_weights_path))

    net = net.to(device)

    if config.train:
        train_paths_10, train_paths_20, train_paths_60 = generate_paths(config.training_img_root, config.training_img_names)
        ds_train = TrainingDataset60m(train_paths_10, train_paths_20, train_paths_60, normalize, input_prepro_60, get_patches, ratio,
                                      config.training_patch_size_60)
        train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)

        if len(config.validation_img_names) != 0:
            val_paths_10, val_paths_20, val_paths_60 = generate_paths(config.validation_img_root, config.validation_img_names)
            ds_val = TrainingDataset60m(val_paths_10, val_paths_20, val_paths_60, normalize, input_prepro_60, get_patches, ratio,
                                        config.training_patch_size_60)
            val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=True)
        else:
            val_loader = None

        net, history = train(device, net, train_loader, config, val_loader)

        if config.save_weights:
            if not os.path.exists(config.save_weights_path):
                os.makedirs(config.save_weights_path)
            torch.save(net.state_dict(), os.path.join(config.save_weights_path, 'DSen2_60m.tar'))

        if config.save_training_stats:
            if not os.path.exists('./Stats/DSen2'):
                os.makedirs('./Stats/DSen2')
            io.savemat('./Stats/DSen2/Training_60m.mat', history)

    bands_high_norm = normalize(bands_high)
    bands_intermediate_lr_norm = normalize(bands_intermediate_lr)
    bands_low_lr_norm = normalize(bands_low_lr)

    patches_10, patches_20, patches_60 = get_test_patches_60(bands_high_norm, bands_intermediate_lr_norm, bands_low_lr_norm, patchSize=config.test_patch_size_60, border=config.border_60)

    output = []

    with torch.no_grad():
        for i in range(len(patches_10)):
            input_10 = patches_10[i].unsqueeze(0).to(device)
            input_20 = patches_20[i].unsqueeze(0).to(device)
            input_60 = patches_60[i].unsqueeze(0).to(device)
            output.append(net(input_10, input_20, input_60))

    output = torch.cat(output, dim=0)
    fused = recompose_images(output, config.border_60, bands_high.shape)
    fused = denormalize(fused)

    torch.cuda.empty_cache()

    return fused.cpu().detach()


def train(device, net, train_loader, config, val_loader=None):

    criterion = torch.nn.L1Loss(reduction='mean').to(device)
    metric = torch.nn.MSELoss(reduction='mean').to(device)
    optim = torch.optim.NAdam(net.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2), eps=config.epsilon, weight_decay=config.schedule_decay)

    history_loss = []
    history_metric = []
    history_val_loss = []
    history_val_metric = []

    pbar = tqdm(range(config.epochs))

    for epoch in pbar:

        pbar.set_description('Epoch %d/%d' % (epoch + 1, config.epochs))
        running_loss = 0.0
        running_metric = 0.0

        running_val_loss = 0.0
        running_val_metric = 0.0

        net.train()

        for i, data in enumerate(train_loader):

            optim.zero_grad()
            if len(data) == 3:
                inputs_10, inputs_20, labels = data
            else:
                inputs_10, inputs_20, inputs_60, labels = data
                inputs_60 = inputs_60.to(device)

            inputs_10 = inputs_10.to(device)
            inputs_20 = inputs_20.to(device)
            labels = labels.to(device)

            if len(data) == 3:
                outputs = net(inputs_10, inputs_20)
            else:
                outputs = net(inputs_10, inputs_20, inputs_60)

            loss = criterion(outputs, labels)
            with torch.no_grad():
                mse = metric(outputs, labels)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            running_metric += mse.item()

        running_loss = running_loss / len(train_loader)
        running_metric = running_metric / len(train_loader)

        if val_loader is not None:
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    if len(data) == 3:
                        inputs_10, inputs_20, labels = data
                        inputs_60 = None
                    else:
                        inputs_10, inputs_20, inputs_60, labels = data
                        inputs_60 = inputs_60.to(device)
                    inputs_10 = inputs_10.to(device)
                    inputs_20 = inputs_20.to(device)
                    labels = labels.to(device)
                    if inputs_60 == None:
                        outputs = net(inputs_10, inputs_20)
                    else:
                        outputs = net(inputs_10, inputs_20, inputs_60)
                    val_loss = criterion(outputs, labels)
                    val_mse = metric(outputs, labels)
                    running_val_loss += val_loss.item()
                    running_val_metric += val_mse.item()

            running_val_loss = running_val_loss / len(val_loader)
            running_val_metric = running_val_metric / len(val_loader)

        history_loss.append(running_loss)
        history_metric.append(running_metric)
        history_val_loss.append(running_val_loss)
        history_val_metric.append(running_val_metric)

        pbar.set_postfix({'loss': running_loss, 'metric': running_metric, 'val_loss': running_val_loss, 'val_metric': running_val_metric})

    history = {'loss': history_loss, 'metric': history_metric, 'val_loss': history_val_loss, 'val_metric': history_val_metric}

    return net, history

