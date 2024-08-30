import os
from scipy import io
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import inspect

from .network import DSen2Model
from .input_preprocessing import normalize, denormalize, get_test_patches_20, get_test_patches_60, recompose_images, upsample_protocol
from Utils.dl_tools import open_config, generate_paths, TrainingDataset20mRR, TrainingDataset60mRR


def DSen2(ordered_dict):
    if ordered_dict.ratio == 2:
        return DSen2_20(ordered_dict)
    else:
        return DSen2_60(ordered_dict)


def DSen2_20(ordered_dict):
    bands_10 = torch.clone(ordered_dict.bands_10)
    bands_20 = torch.clone(ordered_dict.bands_20)

    config_path = os.path.join(os.path.dirname(inspect.getfile(DSen2Model)), 'config.yaml')
    config = open_config(config_path)

    #os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda:" + config.gpu_number if torch.cuda.is_available() else "cpu")

    model_weights_path = config.model_weights_path

    net = DSen2Model((config.number_bands_10, config.number_bands_20))

    if not (config.train and ordered_dict.img_number == 0) or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join(os.path.dirname(inspect.getfile(DSen2Model)), 'weights',
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
            val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False)
        else:
            val_loader = None

        net, history = train(device, net, train_loader, config, val_loader)

        if config.save_weights:
            if not os.path.exists(
                    os.path.join(os.path.dirname(inspect.getfile(DSen2Model)), config.save_weights_path)):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(DSen2Model)), config.save_weights_path))
            torch.save(net.state_dict(),
                       os.path.join(os.path.dirname(inspect.getfile(DSen2Model)), config.save_weights_path,
                                    ordered_dict.dataset + '.tar'))

        if config.save_training_stats:
            if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(DSen2Model)), 'Stats', 'DSen2')):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(DSen2Model)), 'Stats', 'DSen2'))
            io.savemat(
                os.path.join(os.path.dirname(inspect.getfile(DSen2Model)), 'Stats', 'DSen2', 'Training_DSen2_20.mat'),
                history)

    bands_10_norm = normalize(bands_10)
    bands_20_norm = normalize(bands_20)

    patches_10, patches_20 = get_test_patches_20(bands_10_norm, bands_20_norm,
                                                 patch_size=config.test_patch_size_20, border=config.border_20)

    output = []

    with torch.no_grad():
        for i in range(len(patches_10)):
            input_10 = patches_10[i].unsqueeze(0).to(device)
            input_20 = patches_20[i].unsqueeze(0).to(device)
            output.append(net(input_10, input_20))

    output = torch.cat(output, dim=0)
    fused = recompose_images(output, config.border_20, bands_10.shape)
    fused = denormalize(fused)

    torch.cuda.empty_cache()
    return fused.cpu().detach()


def DSen2_60(ordered_dict):
    bands_10 = torch.clone(ordered_dict.bands_10)
    bands_20 = torch.clone(ordered_dict.bands_20)
    bands_60 = torch.clone(ordered_dict.bands_60)

    config_path = os.path.join(os.path.dirname(inspect.getfile(DSen2Model)), 'config.yaml')
    config = open_config(config_path)

    if bands_60.shape[1] > config.number_bands_60:
        bands_60 = bands_60[:, :config.number_bands_60, :, :]

    #os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda:" + config.gpu_number if torch.cuda.is_available() else "cpu")

    model_weights_path = config.model_weights_path

    net = DSen2Model((config.number_bands_10, config.number_bands_20, config.number_bands_60))

    if not (config.train and ordered_dict.img_number == 0) or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join(os.path.dirname(inspect.getfile(DSen2Model)), 'weights',
                                              ordered_dict.dataset + '_60.tar')
        if os.path.exists(model_weights_path):
            net.load_state_dict(torch.load(model_weights_path))
            print('Weights loaded from: ' + model_weights_path)

    net = net.to(device)

    if (config.train or config.resume) and ordered_dict.img_number == 0:
        train_paths = generate_paths(config.training_img_root, 'Reduced_Resolution', 'Training',  '60')
        ds_train = TrainingDataset60mRR(train_paths, normalize)
        train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)

        if len(config.validation_img_names) != 0:
            val_paths = generate_paths(config.training_img_root, 'Reduced_Resolution', 'Validation',  '60')
            ds_val = TrainingDataset60mRR(val_paths, normalize)
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

    bands_10_norm = normalize(bands_10[:, [2, 1, 0, 3], :, :])
    bands_20_norm = normalize(bands_20)
    bands_60_norm = normalize(bands_60)

    patches_10, patches_20, patches_60 = get_test_patches_60(bands_10_norm, bands_20_norm,
                                                             bands_60_norm, patch_size=config.test_patch_size_60,
                                                             border=config.border_60)

    output = []

    with torch.no_grad():
        for i in range(len(patches_10)):
            input_10 = patches_10[i].unsqueeze(0).to(device)
            input_20 = patches_20[i].unsqueeze(0).to(device)
            input_60 = patches_60[i].unsqueeze(0).to(device)
            output.append(net(input_10, input_20, input_60))

    output = torch.cat(output, dim=0)
    fused = recompose_images(output, config.border_60, bands_10.shape)
    fused = denormalize(fused)

    torch.cuda.empty_cache()

    return fused.cpu().detach()


def train(device, net, train_loader, config, val_loader=None):
    criterion = torch.nn.L1Loss(reduction='mean').to(device)
    metric = torch.nn.MSELoss(reduction='mean').to(device)
    optim = torch.optim.NAdam(net.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                              eps=config.epsilon, weight_decay=config.schedule_decay)

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

            inputs_10 = inputs_10[:, [2, 1, 0, 3], :, :].float().to(device)
            inputs_20 = upsample_protocol(inputs_20, inputs_10.shape).float()
            inputs_20 = inputs_20.to(device)
            labels = labels.float().to(device)

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
                        inputs_60 = upsample_protocol(inputs_60, inputs_10.shape).float()
                        inputs_60 = inputs_60.to(device)
                    inputs_10 = inputs_10[:, [2, 1, 0, 3], :, :].float().to(device)
                    inputs_20 = upsample_protocol(inputs_20, inputs_10.shape).float()
                    inputs_20 = inputs_20.to(device)
                    labels = labels.float().to(device)
                    if inputs_60 is None:
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

        pbar.set_postfix({'loss': running_loss, 'metric': running_metric, 'val_loss': running_val_loss,
                          'val_metric': running_val_metric})

    history = {'loss': history_loss, 'metric': history_metric, 'val_loss': history_val_loss,
               'val_metric': history_val_metric}

    return net, history
