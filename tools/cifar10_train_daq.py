import os
import tqdm
import argparse
import pprint
import torch
from tensorboardX import SummaryWriter

from datasets.cifar10 import Cifar10
from models import get_model
from losses import get_loss
from optimizers import get_optimizer, get_q_optimizer
from schedulers import get_scheduler
import utils.config
import utils.checkpoint
from .cifar10_train import evaluate_single_epoch, count_parameters

device = None


def train_single_epoch(model, dataloader, criterion, optimizer, q_optimizer, epoch, writer, postfix_dict):
    model.train()
    total_step = len(dataloader)

    log_dict = {}

    tbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (imgs, labels) in tbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        q_optimizer.zero_grad()

        pred_dict = model(imgs)
        loss = criterion['train'](pred_dict['out'], labels)
        for k, v in loss.items():
            log_dict[k] = v.item()

        loss['loss'].backward()
        optimizer.step()
        q_optimizer.step()

        # logging
        f_epoch = epoch + i / total_step
        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {} epoch'.format(i, total_step, epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)

        # tensorboard
        if i % 10 == 0:
            log_step = int(f_epoch * 1280)
            if writer is not None:
                for key, value in log_dict.items():
                    writer.add_scalar('train/{}'.format(key), value, log_step)


def train(config, model, dataloaders, criterion, optimizer, q_optimizer, scheduler, q_scheduler, writer, start_epoch):
    num_epochs = config.train.num_epochs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    postfix_dict = {'train/lr': 0.0,
                    'train/loss': 0.0,
                    'train/accuracy': 0.0,
                    'test/accuracy': 0.0,
                    'test/loss': 0.0}

    best_accuracy = 0.0

    for epoch in range(start_epoch, num_epochs):
        # train phase
        train_single_epoch(model, dataloaders['train'], criterion, optimizer, q_optimizer,
                           epoch, writer, postfix_dict)

        # test phase
        top1, top5 = evaluate_single_epoch(model, dataloaders['test'], criterion, epoch, writer,
                                           postfix_dict, eval_type='test')
        scheduler.step()
        q_scheduler.step()

        if best_accuracy < top1:
            best_accuracy = top1

        if epoch % config.train.save_model_frequency == 0:
            utils.checkpoint.save_checkpoint(config, model, optimizer, scheduler, q_optimizer,
                                             q_scheduler, None, None, epoch, 0, 'model')

    utils.checkpoint.save_checkpoint(config, model, optimizer, scheduler, q_optimizer,
                                     q_scheduler, None, None, epoch, 0, 'model')

    return {'best_accuracy': best_accuracy}


def qparam_extract(model):
    var = list()
    for m in model._modules:
        if len(model._modules[m]._modules) > 0:
            var = var + qparam_extract(model._modules[m])
        else:
            if hasattr(model._modules[m], 'init'):
                print("qparam: ", list(model._modules[m].parameters())[1:])
                var = var + list(model._modules[m].parameters())[1:]
    return var


def param_extract(model):
    var = list()
    for m in model._modules:
        if len(model._modules[m]._modules) > 0:
            var = var + param_extract(model._modules[m])
        else:
            from models.quantization.daq.daq import DAQConv2d
            if isinstance(model._modules[m], DAQConv2d):
                print("type", type(model._modules[m]))
            if hasattr(model._modules[m], 'init'):
                print("param: ", list(model._modules[m].parameters())[0:1])
                var = var + list(model._modules[m].parameters())[0:1]
            else:
                var = var + list(model._modules[m].parameters())
    return var


def run(config):
    model = get_model(config).to(device)
    print("The number of parameters : %d" % count_parameters(model))
    criterion = get_loss(config)

    q_param = qparam_extract(model, )
    param = param_extract(model)

    optimizer = get_optimizer(config, param)
    q_optimizer = get_q_optimizer(config, q_param)

    # Loading the full-precision model
    if config.model.pretrain.pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(config.model.pretrain.dir)['state_dict']

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('load the pretrained model')

    checkpoint = utils.checkpoint.get_initial_checkpoint(config, model_type)
    last_epoch, step = -1, -1
    print('model from checkpoint: {} last epoch:{}'.format(checkpoint, last_epoch))
    scheduler = get_scheduler(config, optimizer, last_epoch)
    q_scheduler = get_scheduler(config, q_optimizer, last_epoch)

    cifar10 = Cifar10(data_path=config.data.data_path, train_batch_size=config.train.batch_size,
                      eval_batch_size=config.eval.batch_size,
                      num_workers=config.data.num_workers, pin_memory=config.data.pin_memory)
    dataloaders = cifar10.get_dataloader()

    writer = SummaryWriter(config.train['model_dir'])

    train(config, model, dataloaders, criterion, optimizer, q_optimizer,
          scheduler, q_scheduler, writer, last_epoch+1)


def main(args):
    global device
    global model_type
    model_type = 'model'
    import warnings
    warnings.filterwarnings("ignore")

    print('train %s network' % model_type)
    if args.config_file is None:
        raise Exception('no configuration file')

    config = utils.config.load(args.config_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pprint.PrettyPrinter(indent=2).pprint(config)
    utils.prepare_train_directories(config, model_type)
    run(config)

    print('success!')



