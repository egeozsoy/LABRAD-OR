import warnings

warnings.filterwarnings('ignore')
import argparse
from pathlib import Path

import json_tricks as json  # Allows to load integers etc. correctly
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import ORDataset
from scene_graph_prediction.scene_graph_helpers.model.scene_graph_prediction_model import SGPNModelWrapper
from pytorch_lightning.callbacks import ModelCheckpoint


def config_loader(config_path: str):
    config_path = Path('scene_graph_prediction/scene_graph_helpers/configs') / config_path
    with open(config_path, 'r') as f:
        config = json.load(f, ignore_comments=True)
    return config


def find_checkpoint_path(log_dir: str):
    def epoch_int(file_str):
        return int(file_str.split('=')[1].replace('.ckpt', ''))

    log_dir = Path(log_dir)
    checkpoint_folder = log_dir / 'checkpoints'
    checkpoints = sorted(checkpoint_folder.glob('*.ckpt'), key=lambda x: epoch_int(x.name), reverse=True)
    if len(checkpoints) == 0:
        return None
    return checkpoints[0]


def main():
    # 4 Configs with 4 pretrained models:
    # scene_graph_prediction/scene_graph_helpers/configs/visual_only.json.              Checkpoint: 'scene_graph_prediction/scene_graph_helpers/paper_weights/visual_only.ckpt'
    # scene_graph_prediction/scene_graph_helpers/configs/visual_only_with_images.json.  Checkpoint: 'scene_graph_prediction/scene_graph_helpers/paper_weights/visual_only_with_images.ckpt'
    # scene_graph_prediction/scene_graph_helpers/configs/labrad-or.json.                Checkpoint: 'scene_graph_prediction/scene_graph_helpers/paper_weights/labrad-or.ckpt'
    # scene_graph_prediction/scene_graph_helpers/configs/labrad-or_with_images.json.    Checkpoint: 'scene_graph_prediction/scene_graph_helpers/paper_weights/labrad-or_with_images.ckpt'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='example.json', help='configuration file name. Relative path under given path')
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)
    config = config_loader(args.config)
    mode = 'infer'  # can be train/evaluate/infer
    name = args.config.replace('.json', '')
    print(f'Running {name}')

    logger = pl.loggers.TensorBoardLogger('scene_graph_prediction/scene_graph_helpers/logs', name=name, version=0)
    checkpoint_path = find_checkpoint_path(logger.log_dir)
    # (Optionally) Hardcode specific path if requested e.g.:
    # checkpoint_path = 'scene_graph_prediction/scene_graph_helpers/paper_weights/labrad-or_with_images.ckpt'

    if mode == 'train':
        train_dataset = ORDataset(config, 'train', shuffle_objs=True)
        val_dataset = ORDataset(config, 'val')

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                  collate_fn=train_dataset.collate_fn)
        # class_count = torch.zeros(len(train_dataset.relationNames), dtype=torch.long) + 1
        # total_count = 0
        # for elem in tqdm(train_loader, desc='Calculating class weights for history proxy task'):
        #     hot_rels = elem['hot_rels']  # has 0 or 1 per rel
        #     class_count += hot_rels
        #     total_count += 1
        # # Every class occurs class_count times positive, otherwise negative. The ratio of positive to negative is
        # pos_to_neg_ratio = class_count / (total_count - class_count) # If 10 positives, 90 negatives, this is 10/90 = 0.11
        # # The weight for each class is the inverse of the ratio
        # pos_weights = 1 / pos_to_neg_ratio
        # print(f'Pos weights: {pos_weights}')
        # We provide these weights in this repo, but you can also calculate them yourself.
        # torch.save(pos_weights, 'proxy_pos_weights.pth')

        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=val_dataset.collate_fn)

        model = SGPNModelWrapper(config, num_class=len(val_dataset.classNames), num_rel=len(val_dataset.relationNames), weights_obj=train_dataset.w_cls_obj,
                                 weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames)

        checkpoint = ModelCheckpoint(filename='{epoch}', save_top_k=-1, every_n_epochs=1)
        trainer = pl.Trainer(gpus=1, max_epochs=config['MAX_EPOCHES'], logger=logger, check_val_every_n_epoch=1, log_every_n_steps=50, num_sanity_val_steps=0,
                             callbacks=[checkpoint, pl.callbacks.progress.RichProgressBar()], benchmark=False,
                             precision=16)
        print('Start Training')
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)

    elif mode == 'evaluate':
        train_dataset = ORDataset(config, 'train', shuffle_objs=True)
        eval_dataset = ORDataset(config, 'val')
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                 collate_fn=eval_dataset.collate_fn)

        try:
            model = SGPNModelWrapper.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, num_class=len(eval_dataset.classNames),
                                                          num_rel=len(eval_dataset.relationNames), weights_obj=train_dataset.w_cls_obj,
                                                          weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames)
        except:
            print('Strict loading failed, trying non-strict')
            model = SGPNModelWrapper(config, num_class=len(eval_dataset.classNames), num_rel=len(eval_dataset.relationNames),
                                     weights_obj=train_dataset.w_cls_obj,
                                     weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames)
            print(model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=False))
            checkpoint_path = None

        trainer = pl.Trainer(gpus=1, max_epochs=config['MAX_EPOCHES'], logger=logger, check_val_every_n_epoch=1, log_every_n_steps=50, num_sanity_val_steps=0,
                             callbacks=[pl.callbacks.progress.RichProgressBar()])
        trainer.validate(model, eval_loader, ckpt_path=checkpoint_path)
    elif mode == 'infer':
        train_dataset = ORDataset(config, 'train', shuffle_objs=True)
        infer_split = 'test'
        eval_dataset = ORDataset(config, infer_split, for_eval=True)
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                 collate_fn=eval_dataset.collate_fn)

        try:
            model = SGPNModelWrapper.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, num_class=len(eval_dataset.classNames),
                                                          num_rel=len(eval_dataset.relationNames), weights_obj=train_dataset.w_cls_obj,
                                                          weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames)
        except:
            print('Strict loading failed, trying non-strict')
            model = SGPNModelWrapper(config, num_class=len(eval_dataset.classNames), num_rel=len(eval_dataset.relationNames),
                                     weights_obj=train_dataset.w_cls_obj,
                                     weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames)
            print(model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=False))
            checkpoint_path = None

        trainer = pl.Trainer(gpus=1, max_epochs=config['MAX_EPOCHES'], logger=logger, check_val_every_n_epoch=1, log_every_n_steps=50, num_sanity_val_steps=0,
                             callbacks=[pl.callbacks.progress.RichProgressBar()])
        results = trainer.predict(model, eval_loader, ckpt_path=checkpoint_path)
        scan_relations = {key: value for key, value in results}
        output_name = f'scan_relations_{name}_{infer_split}.json'
        with open(output_name, 'w') as f:
            json.dump(scan_relations, f)
    else:
        raise ValueError('Invalid mode')


if __name__ == '__main__':
    import subprocess

    subprocess.call(['nvidia-smi', '-L'])
    main()
