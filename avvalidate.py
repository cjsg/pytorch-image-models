#!/usr/bin/env python3
""" Adversarial Validation Script

This is intended for adversarial validation on ImageNet and CIFAR-10 of pretrained networks.  It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Heavily based on timm's validation.py script from Ross Wightman's timm package (https://github.com/rwightman),
which stands under Apache 2.0 license.

Modified by Carl-Johann Simon-Gabriel.
"""
import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models, NormalizationWrapper
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy

import foolbox as fb
from pathlib import Path

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset. '
                         'If path contains `lmdb`, then LMDBIterDataset/Loader is used.')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-interval', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=None,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--buffer-size', type=int, default=None, metavar='B',
                    help='buffer size when using iterable lmdb-dataset')

# Adversarial attacks
parser.add_argument('--attacks', type=str, default=None, nargs='*', choices=[None, 'fgm', 'fgsm', 'l2pgd', 'linfpgd'],
                    help='Should be fgsm or pgd. Default=None.')
parser.add_argument('--attack-sizes', type=float, default=0., nargs='*',
                    help='Size of the attack. Attack must be not None. NB: Both normalization and resizing '
                    'are part of the model. Attack-size refers to the standard img size (32 for cifar, 224 for imagenet).')
parser.add_argument('--attack-random-start', type=bool, default=True,
                    help='Wheter to use a random point in the epsilong ball to start the attac.k')
parser.add_argument('--attack-steps', type=int, default=7,
                    help='How many PGD steps to use per attack. Attack must be l2pgd or linfpgd.')
parser.add_argument('--no-images', action='store_true', default=False,
                    help='do not save a batch of adversarial images')


def validate(args, attack=None):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.save_imgs = not args.no_images
    if args.device == 'cpu' and not args.no_prefetcher:
        args.no_prefetcher = True
        _logger.warning("Pre-fetcher needs GPUs. I will switch on the argument --no-prefetcher.")
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp or args.apex_amp or args.native_amp:
        assert args.device == 'cuda', 'You need GPUs when enabling --amp'
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    num_classes = args.num_classes
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    if num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        # TODO: implement a default number of classes = 10 for cifar and = 1000 for imgnet
        # Currently, jx_nest_tiny and vitB16 got fine-tuned on cifar but using 1000 class output
        num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)

    # if adversarial training, include normalization layer into the model:
    # models' vulnerability must be compared w.r.t. [0, 1] scaled inputs
    model = NormalizationWrapper(
        model, data_config['mean'], data_config['std'], args.channels_last,
        data_config['input_size'], data_config['interpolation'])  # include resizing into net
    model.is_model_wrapper = True
    data_config['mean'] = tuple([0.] * len(data_config['mean']))
    data_config['std'] = tuple([1.] * len(data_config['std']))
    _logger.info(
        'BEWARE: if img resizing is needed it gets integrated into the model '
        'via a wrapper. Hence the input-size of images is always 32 for cifar '
        'and 224 for imgnet, and the attack is conducted on that input. Hence '
        '`attack_size` refers to img-size 32 for cifar and 224 for imgnet.')
    if 'cifar' in args.data.lower():
        data_config['input_size'] = (32, 32, 3) if args.channels_last else (3, 32, 32)
    elif 'imagenet' in args.data.lower() or 'imgnet' in args.data.lower():
        data_config['input_size'] = (224, 224, 3) if args.channels_last else (3, 224, 224)
    else:
        raise NotImplementedError(f'Unknown default input_size for dataset {args.data}')

    test_time_pool = False
    if not args.no_test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.to(args.device)
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    dataset = create_dataset(
        name=args.dataset, root=args.data, split=args.split,
        load_bytes=args.tf_preprocessing, class_map=args.class_map, is_training=False)

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(num_classes)]
    else:
        valid_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing)

    # create adv attack
    fmodel = None
    random_start = args.attack_random_start
    if attack is None:
        assert len(args.attack_sizes) == 1 and args.attack_sizes[0] == 0.
        attack = fb.attacks.FGM(random_start=False)  # use FGM with epsilon = 0. and no random start
    elif attack == 'fgm':
        attack = fb.attacks.FGM(random_start=random_start)
    elif attack == 'fgsm':
        attack = fb.attacks.FGSM(random_start=random_start)
    elif attack == 'l2pgd':
        attack = fb.attacks.L2PGD(steps=args.attack_steps, random_start=random_start)
    elif attack == 'linfpgd':
        attack = fb.attacks.LinfPGD(steps=args.attack_steps, random_start=random_start)
    else:
        raise ValueError(f'Unknown attack {attack}')
    model.eval()
    fmodel = fb.PyTorchModel(model, bounds=(0., 1.), device=args.device)


    batch_time = AverageMeter()
    accs = AverageMeter()

    model.eval()
    # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
    input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).to(args.device)
    if args.channels_last:
        input = input.contiguous(memory_format=torch.channels_last)
    model(input)
    end = time.time()
    saved_imgs = None
    for batch_idx, (input, target) in enumerate(loader):
        if args.no_prefetcher:
            target = target.to(args.device)
            input = input.to(args.device)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # if attack is not None:
        assert not args.channels_last
        model.eval()
        hard_target = target.argmax(dim=-1) if len(target.shape) > 1 else target  # TODO: use true/soft labels, not hard labels
        with amp_autocast():
            _, advs, success_ = attack(fmodel, input, hard_target, epsilons=args.attack_sizes, check_success=True)
        assert success_.shape == (len(args.attack_sizes), input.size(0))
        model.zero_grad()

        success = success_.detach()
        acc = (~success).sum(-1).cpu().numpy() / input.size(0) * 100.
        accs.update(acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx == 0 and args.save_imgs:
            saved_imgs = [adv[:16].detach().cpu().numpy() for adv in advs]

        if batch_idx % args.log_interval == 0:
            rate_avg=input.size(0) / batch_time.avg
            log_str = f'[{batch_idx:>4d}/{len(loader)}]  '\
                      f'{batch_time.val:>6.3f}s ({batch_time.avg:>6.3f}s, {rate_avg:>7.2f}/s)  '
            acc_str = '  '.join([
                f'{acc_val:>7.3f} ({acc_avg:>7.3f})' for (acc_val, acc_avg) in zip(accs.val, accs.avg)])
            if batch_idx == 0:  # print column titles
                columns = ' '*len(log_str) + ''.join([' '*4 + f'Acc@{eps:3.1e}' + ' '*4 for eps in args.attack_sizes])
                _logger.info(columns)
            _logger.info( f'{log_str}{acc_str}')

    results = OrderedDict(
        epsilons=args.attack_sizes,
        accs=list(accs.avg),
        errors=list(100.-accs.avg),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        cropt_pct=crop_pct,
        interpolation=data_config['interpolation'],
        images=saved_imgs)

    eps_str = ''.join([' '*5 + f'@{eps:.1e}' + ' '*6 for eps in args.attack_sizes])
    acc_str = '  '.join([f'{acc:>7.3f} ({err:>7.3f})' for (acc, err) in zip(results['accs'], results['errors'])])
    _logger.info(' * Attack Size:   ' + eps_str)
    _logger.info('   Accs (Errors): ' + acc_str)

    return results


def main():
    setup_default_logging()
    args = parser.parse_args()
    if not isinstance(args.attacks, (tuple, list)):
        assert type(args.attacks) == str
        args.attacks = [args.attacks]
    if not isinstance(args.attack_sizes, (tuple, list)):
        assert type(args.attack_sizes) == float
        args.attack_sizes = [args.attack_sizes]

    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]
        else:
            model_names = [args.model]
            model_cfgs = [(args.model, args.checkpoint)]

    if torch.cuda.is_available():
        args.device = 'cuda'
        if args.num_gpu is None:
            args.num_gpu = 1
    else:
        args.device = 'cpu'
        if args.num_gpu is None:
            args.num_gpu = 0
        assert args.num_gpu == 0, '--num-gpu != 0, but no GPU was found'


    if len(model_cfgs):
        if args.results_file:
            results_file = args.results_file
        elif args.checkpoint and os.path.isdir(args.checkpoint):
            results_file = os.path.join(args.checkpoint, 'results-test.pth')
        elif args.checkpoint:
            results_file = os.path.join(Path(args.checkpoint).parent.absolute(), 'results-test.pth')
        else:
            results_file = './results-test.pth'
            
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                for attack in args.attacks:
                    batch_size = start_batch_size
                    args.model = m
                    args.checkpoint = c
                    result = OrderedDict(model=args.model, attack=attack)
                    r = {}
                    while not r and batch_size >= args.num_gpu:
                        torch.cuda.empty_cache()
                        try:
                            args.batch_size = batch_size
                            print('Validating with batch size: %d' % args.batch_size)
                            r = validate(args, attack)
                        except RuntimeError as e:
                            if batch_size <= args.num_gpu:
                                print("Validation failed with no ability to reduce batch size. Exiting.")
                                raise e
                            batch_size = max(batch_size // 2, args.num_gpu)
                            print("Validation failed, reducing batch size by 50%")
                    result.update(r)
                    if args.checkpoint:
                        result['checkpoint'] = args.checkpoint
                    results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['accs'][0], reverse=True)
        if len(results) > 1:
            torch.save(results, results_file)
        elif len(results) == 1:
            torch.save(results[0], results_file)
    else:
        for attack in args.attacks:
            validate(args, attack=attack)


# def write_results(results_file, results):
#     with open(results_file, mode='w') as cf:
#         dw = csv.DictWriter(cf, fieldnames=results[0].keys())
#         dw.writeheader()
#         for r in results:
#             dw.writerow(r)
#         cf.flush()


if __name__ == '__main__':
    main()
