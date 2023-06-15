'''
This should easily support only processing 2D, 3D, both, parts or full, using different encoders, inputs, etc.

'''
import random
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from datetime import datetime
from itertools import chain
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertModel

from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import convert_nodes_edges_to_graph
from scene_graph_prediction.scene_graph_helpers.model.gcns.network_TripletGCN import TripletGCNModel
from scene_graph_prediction.scene_graph_helpers.model.graphormer import collator
from scene_graph_prediction.scene_graph_helpers.model.graphormer.model import Graphormer
from scene_graph_prediction.scene_graph_helpers.model.model_utils import get_image_model
from scene_graph_prediction.scene_graph_helpers.model.pointnets.network_PointNet import PointNetCls, PointNetRelCls
from scene_graph_prediction.scene_graph_helpers.model.pointnets.network_PointNet2 import PointNetfeat as PointNetfeat2


class SGPNModelWrapper(pl.LightningModule):
    def __init__(self, config, num_class, num_rel, weights_obj, weights_rel, relationNames):
        super().__init__()
        self.config = config
        self.mconfig = config['MODEL']
        self.n_object_types = 6
        self.weights_obj = weights_obj
        self.weights_rel = weights_rel
        self.relationNames = relationNames
        self.lr = float(self.config['LR'])
        # evaluation metrics
        self.train_take_rel_preds = defaultdict(list)
        self.train_take_rel_gts = defaultdict(list)
        self.val_take_rel_preds = defaultdict(list)
        self.val_take_rel_gts = defaultdict(list)

        self.train_take_rel_gts_proxy = defaultdict(list)
        self.train_take_rel_preds_proxy = defaultdict(list)
        self.val_take_rel_gts_proxy = defaultdict(list)
        self.val_take_rel_preds_proxy = defaultdict(list)

        self.reset_metrics()

        self.obj_encoder = PointNetfeat2(input_dim=6, out_size=self.mconfig['point_feature_size'], input_dropout=self.mconfig['INPUT_DROPOUT'])
        self.rel_encoder = PointNetfeat2(input_dim=7, out_size=self.mconfig['edge_feature_size'], input_dropout=self.mconfig['INPUT_DROPOUT'])
        if self.config['IMAGE_INPUT'] == 'full':
            self.full_image_model, _ = get_image_model(model_config=self.mconfig)
            # Freeze the whole model
            for param in self.full_image_model.parameters():
                param.requires_grad = False
            # Unfreeze conv head
            for param in chain(self.full_image_model.conv_head.parameters()):
                param.requires_grad = True
            self.full_image_feature_reduction = nn.Linear(self.full_image_model.num_features, self.mconfig['FULL_IMAGE_EMBEDDING_SIZE'] // 6)

        self.gcn = TripletGCNModel(num_layers=self.mconfig['N_LAYERS'],
                                   dim_node=self.mconfig['point_feature_size'],
                                   dim_edge=self.mconfig['edge_feature_size'],
                                   dim_hidden=self.mconfig['gcn_hidden_feature_size'])

        # node feature classifier
        self.obj_predictor = PointNetCls(num_class, in_size=self.mconfig['point_feature_size'],
                                         batch_norm=False, drop_out=True)
        rel_in_size = self.mconfig['edge_feature_size']
        self.rel_predictor = PointNetRelCls(
            num_rel,
            in_size=rel_in_size,
            batch_norm=False, drop_out=True, image_embedding_size=self.mconfig['FULL_IMAGE_EMBEDDING_SIZE'] if self.config['IMAGE_INPUT'] == 'full' else None,
            n_object_types=self.n_object_types,
            history_embedding_size=self.mconfig['HISTORY_DIM'] * 2 if self.config['USE_HISTORY'] else None)

        self.history_dim = self.mconfig['HISTORY_DIM']
        if self.config['USE_HISTORY']:
            self.sg_model = Graphormer(n_layers=self.mconfig['N_GRAPH_LAYERS'], hidden_dim=self.history_dim)
            self.history_model_config = BertConfig(vocab_size=30, hidden_size=self.history_dim, num_hidden_layers=self.mconfig['N_HISTORY_LAYERS'],
                                                   num_attention_heads=self.mconfig['N_ATTN_HEADS'],
                                                   intermediate_size=self.history_dim * 4, max_position_embeddings=1600)
            self.history_model = BertModel(self.history_model_config)
            self.long_term_history_model = BertModel(self.history_model_config)
            self.short_term_history_model = BertModel(self.history_model_config)
            self.hot_rels_fc = nn.Linear(self.history_dim * 2, num_rel)  # *2 for longshort term

        self.predicate_order = {'Cementing': 0, 'Cleaning': 1, 'Cutting': 0, 'Drilling': 0, 'Hammering': 0, 'Operating': 1, 'Preparing': 0,
                                'Sawing': 0, 'Suturing': 0, 'Touching': 2}

        self.proxy_pos_weight = torch.load('proxy_pos_weights.pth')

        pretrained_state_dict = {}
        if self.mconfig['PRETRAINED_VISUAL_WEIGHTS'] is not None:
            pretrained_state_dict = torch.load(self.mconfig['PRETRAINED_VISUAL_WEIGHTS'])['state_dict']
            keys_to_ignore = ['rel_predictor.fc3.weight', 'sg_model', 'history_model']
            for key in list(pretrained_state_dict.keys()):
                if any([k in key for k in keys_to_ignore]):
                    del pretrained_state_dict[key]

        info = self.load_state_dict(pretrained_state_dict, strict=False)
        # optionally print(info)
        # # Freeze obj_encoder, rel_encoder, gcn, obj_predictor, rel_predictor (not the last fc layer, called fc3)
        # for param in self.get_model_parameters(['visual']):
        #     param.requires_grad = False
        # for param in self.rel_predictor.fc3.parameters():
        #     param.requires_grad = True

        self.register_buffer('relevant_token_tensor', torch.tensor(20, dtype=torch.long))
        self.register_buffer('unknown_token_tensor', torch.tensor(21, dtype=torch.long))
        self.prev_history_per_take = defaultdict(list)

    def get_model_parameters(self, types: List[str]):
        '''
        types: ['visual','history','sg']
        Mainly used for freezing parameters
        '''
        params = []
        if 'visual' in types:
            params.extend(self.obj_encoder.parameters())
            params.extend(self.rel_encoder.parameters())
            params.extend(self.gcn.parameters())
            params.extend(self.obj_predictor.parameters())
            params.extend(self.rel_predictor.parameters())
            if self.config['IMAGE_INPUT'] == 'full':
                params.extend(self.full_image_model.parameters())
        if 'history' in types:
            params.extend(self.history_model.parameters())
            params.extend(self.long_term_history_model.parameters())
            params.extend(self.short_term_history_model.parameters())
            params.extend(self.hot_rels_fc.parameters())
        if 'sg' in types:
            params.extend(self.sg_model.parameters())
        return params

    def freeze_image_model_batchnorm(self):
        models_to_freeze = []
        if self.config['IMAGE_INPUT'] == 'full':
            models_to_freeze.append(self.full_image_model)
        for image_model in models_to_freeze:
            for module in image_model.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()

    def forward(self, batch, history_sg=None, return_meta_data=False, is_train=False, ignore_history=False):
        history_features = None
        proxy_task_pred = None
        history_idx = batch['scan_idx']
        if self.config['USE_HISTORY'] and not ignore_history:
            if history_sg is not None:
                history_sg = history_sg[:history_idx]  # See everything till current token, but not the current token
                if len(history_sg) == 0:
                    history_features = torch.empty(0, self.sg_model.hidden_dim, device=self.sg_model.device, dtype=self.sg_model.dtype)
                else:
                    history_features = self.sg_model(history_sg)
            else:
                raise NotImplementedError

            # mark relevant location in history
            relevant_token_feature = self.history_model.embeddings.word_embeddings(self.relevant_token_tensor)
            unknown_token_feature = self.history_model.embeddings.word_embeddings(self.unknown_token_tensor)
            if history_idx < len(history_features):
                history_features[history_idx] = relevant_token_feature  # replace the current token with the relevant token
            else:
                history_features = torch.cat([history_features, relevant_token_feature.unsqueeze(0)], dim=0)  # add the relevant token at the end
            if is_train and self.config['AUGMENT_HISTORY']:
                # History Augmentations are important to create diversity
                # Goals: Both the content, and the length should change
                # We differentiate between Long-Term and Short-Term History, and augment them seperately, to make sure that the model can learn to use both
                augs = {'remove': False}
                p = random.random()
                history_len = len(history_features)
                if p < 0.2:  # We sometimes(20% of the time) remove long term history
                    augs['remove'] = 'long-term'
                elif p < 0.4:  # We sometimes(20% of the time) remove short term history
                    augs['remove'] = 'short-term'
                # 1. Remove Long Term, defined as anything N steps or more away from the current step in both directions, where N is randomly selected between 20 and 200)
                if augs['remove'] == 'long-term':
                    long_term_N = random.randint(20, 200)
                    augs['previous_remove_end'] = max(0, history_idx - long_term_N)
                    augs['future_remove_start'] = min(history_len, history_idx + long_term_N)
                    # remove previous history
                    history_features[:augs['previous_remove_end']] = unknown_token_feature
                    # remove future history
                    history_features[augs['future_remove_start']:] = unknown_token_feature

                # 2. Remove Short Term, defined as anything N steps or less from the current step in both directions, where N is randomly selected between 1 and 20)
                elif augs['remove'] == 'short-term':
                    short_term_N = random.randint(1, 20)
                    augs['previous_remove_start'] = max(0, history_idx - short_term_N)
                    augs['future_remove_end'] = min(history_len, history_idx + short_term_N)
                    # remove previous history
                    history_features[augs['previous_remove_start']:history_idx] = unknown_token_feature
                    # remove future history
                    history_features[history_idx + 1:augs['future_remove_end']] = unknown_token_feature

                history_features[history_idx] = relevant_token_feature

            history_idx = torch.all(torch.eq(history_features, relevant_token_feature), dim=1).nonzero()[0].item()
            # get every N.th token but make sure the crop includes the relevant token
            timepoint_keys = list(range(len(history_features)))
            long_term_history_features = history_features[history_idx % self.mconfig["LONG_HISTORY_STRIDE"]::self.mconfig["LONG_HISTORY_STRIDE"]]
            long_term_timepoint_keys = timepoint_keys[history_idx % self.mconfig["LONG_HISTORY_STRIDE"]::self.mconfig["LONG_HISTORY_STRIDE"]]
            long_term_history_relevant_idx = torch.all(torch.eq(long_term_history_features, relevant_token_feature), dim=1).nonzero()[0].item()
            short_term_history_features = history_features[max(0, history_idx - self.mconfig["SHORT_HISTORY_LENGTH"]):history_idx + self.mconfig[
                "SHORT_HISTORY_LENGTH"] + 1]
            short_term_timepoint_keys = timepoint_keys[max(0, history_idx - self.mconfig["SHORT_HISTORY_LENGTH"]):history_idx + self.mconfig[
                "SHORT_HISTORY_LENGTH"] + 1]
            short_term_history_relevant_idx = torch.all(torch.eq(short_term_history_features, relevant_token_feature), dim=1).nonzero()[0].item()

            # positional ids are centered around the relevant token, while avoding negative ids. This is very important for performance and generalization.
            long_term_pos_ids = torch.arange(len(long_term_history_features), device=long_term_history_features.device) \
                                - long_term_history_relevant_idx + self.history_model_config.max_position_embeddings // 2
            short_term_pos_ids = torch.arange(len(short_term_history_features), device=short_term_history_features.device) \
                                 - short_term_history_relevant_idx + self.history_model_config.max_position_embeddings // 2

            long_term_history_features = self.long_term_history_model(inputs_embeds=long_term_history_features[None], position_ids=long_term_pos_ids,
                                                                      output_attentions=True)
            short_term_history_features = self.short_term_history_model(inputs_embeds=short_term_history_features[None], position_ids=short_term_pos_ids,
                                                                        output_attentions=True)
            history_features = torch.cat([long_term_history_features.last_hidden_state[0, long_term_history_relevant_idx],
                                          short_term_history_features.last_hidden_state[0, short_term_history_relevant_idx]], dim=0)
            proxy_task_pred = self.hot_rels_fc(history_features)

        obj_feature = self.obj_encoder(batch['obj_points'])
        rel_feature = self.rel_encoder(batch['rel_points'])
        probs = None
        gcn_obj_feature, gcn_rel_feature = self.gcn(obj_feature, rel_feature, batch['edge_indices'])

        if self.mconfig['OBJ_PRED_FROM_GCN']:
            obj_cls = self.obj_predictor(gcn_obj_feature)
        else:
            obj_cls = self.obj_predictor(obj_feature)
        if self.config['IMAGE_INPUT'] == 'full':
            self.freeze_image_model_batchnorm()
            image_features = self.full_image_model(batch['full_image'])
            image_features = self.full_image_feature_reduction(image_features).flatten()
            rel_cls = self.rel_predictor(gcn_rel_feature, relation_objects_one_hot=batch['relation_objects_one_hot'], image_embeddings=image_features,
                                         history_embeddings=history_features)
        else:
            rel_cls = self.rel_predictor(gcn_rel_feature, relation_objects_one_hot=batch['relation_objects_one_hot'], history_embeddings=history_features)

        if return_meta_data:
            return obj_cls, rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs, proxy_task_pred
        else:
            return obj_cls, rel_cls, proxy_task_pred

    def reset_metrics(self, split=None):
        if split == 'train':
            self.train_take_rel_preds = defaultdict(list)
            self.train_take_rel_gts = defaultdict(list)
            self.train_take_rel_preds_proxy = defaultdict(list)
            self.train_take_rel_gts_proxy = defaultdict(list)
        elif split == 'val':
            self.val_take_rel_preds = defaultdict(list)
            self.val_take_rel_gts = defaultdict(list)
            self.val_take_rel_preds_proxy = defaultdict(list)
            self.val_take_rel_gts_proxy = defaultdict(list)
        else:
            self.train_take_rel_preds = defaultdict(list)
            self.train_take_rel_gts = defaultdict(list)
            self.val_take_rel_preds = defaultdict(list)
            self.val_take_rel_gts = defaultdict(list)
            self.train_take_rel_preds_proxy = defaultdict(list)
            self.train_take_rel_gts_proxy = defaultdict(list)
            self.val_take_rel_preds_proxy = defaultdict(list)
            self.val_take_rel_gts_proxy = defaultdict(list)

    def update_metrics(self, batch, rel_pred, split='train'):
        if split == 'train':
            self.train_take_rel_preds[batch['take_idx']].extend(rel_pred.detach().argmax(1).cpu().numpy())
            self.train_take_rel_gts[batch['take_idx']].extend(batch['gt_rels'].detach().cpu().numpy())
        elif split == 'val':
            self.val_take_rel_preds[batch['take_idx']].extend(rel_pred.detach().argmax(1).cpu().numpy())
            self.val_take_rel_gts[batch['take_idx']].extend(batch['gt_rels'].detach().cpu().numpy())
        else:
            raise NotImplementedError()

    def update_proxy_metrics(self, batch, hot_rel_pred_hard, gt_hot_rels, split='train'):
        if split == 'train':
            self.train_take_rel_preds_proxy[batch['take_idx']].append(hot_rel_pred_hard.detach().cpu().tolist())
            self.train_take_rel_gts_proxy[batch['take_idx']].append(gt_hot_rels.detach().cpu().tolist())
        elif split == 'val':
            self.val_take_rel_preds_proxy[batch['take_idx']].append(hot_rel_pred_hard.detach().cpu().tolist())
            self.val_take_rel_gts_proxy[batch['take_idx']].append(gt_hot_rels.detach().cpu().tolist())
        else:
            raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        obj_pred, rel_pred, _, _, _, _, probs, proxy_task_pred = self(batch, history_sg=batch['gt_sgs'] if 'gt_sgs' in batch else None, is_train=True, return_meta_data=True)
        loss_obj = F.nll_loss(obj_pred, batch['gt_class'], weight=self.weights_obj.to(batch['gt_class'].device))
        loss_rel = F.nll_loss(rel_pred, batch['gt_rels'], weight=self.weights_rel.to(batch['gt_rels'].device))
        self.update_metrics(batch, rel_pred, split='train')

        if proxy_task_pred is not None:
            loss_proxy = F.binary_cross_entropy_with_logits(proxy_task_pred, batch['hot_rels'].to(proxy_task_pred), pos_weight=self.proxy_pos_weight.to(proxy_task_pred.device))
            hot_rel_pred_hard = (proxy_task_pred.sigmoid() > 0.5).float()
            loss = self.mconfig['lambda_o'] * loss_obj + loss_rel + loss_proxy
            self.update_proxy_metrics(batch, hot_rel_pred_hard, batch['hot_rels'], split='train')
        else:
            loss = self.mconfig['lambda_o'] * loss_obj + loss_rel

        if batch_idx % 100 == 0:
            print(f'{datetime.now()}: Training step: {batch_idx} ---- Loss: {loss.item()}')
        return loss

    def validation_step(self, batch, batch_idx):
        obj_pred, rel_pred, _, _, _, _, probs, proxy_task_pred = self(batch, history_sg=batch['gt_sgs'] if 'gt_sgs' in batch else None, is_train=False, return_meta_data=True)
        loss_obj = F.nll_loss(obj_pred, batch['gt_class'], weight=self.weights_obj.to(batch['gt_class'].device))
        loss_rel = F.nll_loss(rel_pred, batch['gt_rels'], weight=self.weights_rel.to(batch['gt_rels'].device))
        self.update_metrics(batch, rel_pred, split='val')

        if proxy_task_pred is not None:
            loss_proxy = F.binary_cross_entropy_with_logits(proxy_task_pred, batch['hot_rels'].to(proxy_task_pred), pos_weight=self.proxy_pos_weight.to(proxy_task_pred.device))
            hot_rel_pred_hard = (proxy_task_pred.sigmoid() > 0.5).float()
            loss = self.mconfig['lambda_o'] * loss_obj + loss_rel + loss_proxy
            self.update_proxy_metrics(batch, hot_rel_pred_hard, batch['hot_rels'], split='val')
        else:
            loss = self.mconfig['lambda_o'] * loss_obj + loss_rel

        if batch_idx % 10 == 0:
            print(f'{datetime.now()}: Validation step: {batch_idx} ---- Loss: {loss.item()}')
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.config['USE_HISTORY']:
            print(f'Infering with history')
            if len(self.prev_history_per_take[batch['take_idx']]) > 0:
                history = collator.collator([elem['sg'] for elem in self.prev_history_per_take[batch['take_idx']]]).to(self.device)
            else:
                history = []
            obj_pred, rel_pred, _, _, _, _, probs, _ = self(batch, history_sg=history, return_meta_data=True)
        else:
            obj_pred, rel_pred, _, _, _, _, probs, _ = self(batch, return_meta_data=True)

        predicted_relations = torch.max(rel_pred.detach(), 1)[1]
        all_scores = F.softmax(rel_pred, dim=1)

        # Get the scores that correspond to predicted_relations
        # scores = all_scores[range(rel_pred.shape[0]), predicted_relations]
        relations = []
        for idy, (edge, rel) in enumerate(zip(batch['edge_indices'].transpose(0, 1), predicted_relations)):
            if rel == self.relationNames.index('none'):
                continue
            start = edge[0]
            end = edge[1]
            start_name = batch['objs_json'][start.item() + 1]
            end_name = batch['objs_json'][end.item() + 1]
            rel_name = self.relationNames[rel]
            # print(f'{start_name} -> {rel_name} -> {end_name}')
            # if output_scores: relations.append((start_name, rel_name, end_name, scores[idy].item()))
            relations.append((start_name, rel_name, end_name))

        if self.config['USE_HISTORY']:
            # map indices to human-readable names both for objects and relationships
            nodes = set()
            for sub, rel, obj in relations:
                nodes.add(sub)
                nodes.add(obj)
                # rel is also a node
                nodes.add(f'$_{rel}')
            nodes = sorted(nodes)
            edges = []
            for sub, rel, obj in relations:
                sub_node = nodes.index(sub)
                obj_node = nodes.index(obj)
                rel_node = nodes.index(f'$_{rel}')
                edges.append((sub_node, rel_node, rel))
                edges.append((rel_node, obj_node, rel))

            data = convert_nodes_edges_to_graph(nodes, edges)
            self.prev_history_per_take[batch['take_idx']].append({'scan_id': int(batch['scan_id']), 'sg': collator.preprocess_item(data), 'sg_rels': relations})

        return (batch['scan_id'], relations)

    # def test_step(self, batch, batch_idx): # not for inference
    #     return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        loss = sum(([i['loss'] for i in outputs]))
        self.evaluate_predictions(loss, 'train')
        self.evaluate_proxy_predictions(loss, 'train')
        self.reset_metrics(split='train')

    def validation_epoch_end(self, outputs):
        loss = sum(outputs)
        self.evaluate_predictions(loss, 'val')
        self.evaluate_proxy_predictions(loss, 'val')
        self.reset_metrics(split='val')

        if self.config['USE_HISTORY']:
            if self.trainer.current_epoch % 5 == 0 or self.trainer.current_epoch == self.trainer.max_epochs - 1:
                if self.trainer.train_dataloader is not None:
                    self.evaluate_with_pred_sg(self.trainer.train_dataloader.dataset.datasets, 'train')
                if self.trainer.val_dataloaders is not None:
                    self.evaluate_with_pred_sg(self.trainer.val_dataloaders[0].dataset, 'val')

    # def test_epoch_end(self, outputs):
    #     return self.validation_epoch_end(outputs)

    def evaluate_predictions(self, epoch_loss, split):
        if split == 'train':
            take_rel_preds = self.train_take_rel_preds
            take_rel_gts = self.train_take_rel_gts
        elif split == 'val':
            take_rel_preds = self.val_take_rel_preds
            take_rel_gts = self.val_take_rel_gts
        else:
            raise NotImplementedError()

        self.log(f'Epoch_Loss/{split}', epoch_loss)
        all_rel_gts = []
        all_rel_preds = []
        for take_idx in sorted(take_rel_preds.keys()):
            rel_preds = take_rel_preds[take_idx]
            rel_gts = take_rel_gts[take_idx]
            all_rel_gts.extend(rel_gts)
            all_rel_preds.extend(rel_preds)
            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(self.relationNames))),
                                               target_names=self.relationNames, output_dict=True)
            for rel_name in self.relationNames:
                for score_type in ['precision', 'recall', 'f1-score']:
                    self.log(f'{rel_name}/{take_idx}_{score_type[:2].upper()}', cls_report[rel_name][score_type])

            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(self.relationNames))),
                                               target_names=self.relationNames)
            print(f'\nTake {take_idx}\n')
            print(cls_report)

        results = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                        target_names=self.relationNames, output_dict=True)
        macro_f1 = results['macro avg']['f1-score']
        self.log(f'Epoch_Macro/{split}_PREC', results['macro avg']['precision'])
        self.log(f'Epoch_Macro/{split}_REC', results['macro avg']['recall'])
        self.log(f'Epoch_Macro/{split}_F1', results['macro avg']['f1-score'])
        self.log(f'Epoch_Micro/{split}_PREC', results['weighted avg']['precision'])
        self.log(f'Epoch_Micro/{split}_REC', results['weighted avg']['recall'])
        self.log(f'Epoch_Micro/{split}_F1', results['weighted avg']['f1-score'])
        print(f'{split} Results:\n')
        cls_report = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                           target_names=self.relationNames)
        self.logger.experiment.add_text(f'Classification_Report/{split}', cls_report, self.current_epoch)
        print(cls_report)
        return macro_f1

    def evaluate_proxy_predictions(self, epoch_loss, split):
        if split == 'train':
            take_rel_preds = self.train_take_rel_preds_proxy
            take_rel_gts = self.train_take_rel_gts_proxy
        elif split == 'val':
            take_rel_preds = self.val_take_rel_preds_proxy
            take_rel_gts = self.val_take_rel_gts_proxy
        else:
            raise NotImplementedError()
        # Here we evaluate the proxy prediction
        self.log(f'Epoch_Proxy_Loss/{split}', epoch_loss)
        all_rel_gts = []
        all_rel_preds = []
        for take_idx in sorted(take_rel_preds.keys()):
            rel_preds = take_rel_preds[take_idx]
            rel_gts = take_rel_gts[take_idx]
            all_rel_gts.extend(rel_gts)
            all_rel_preds.extend(rel_preds)
            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(self.relationNames))),
                                               target_names=self.relationNames, output_dict=True)
            for rel_name in self.relationNames:
                for score_type in ['precision', 'recall', 'f1-score']:
                    self.log(f'Proxy_{rel_name}/{take_idx}_{score_type[:2].upper()}', cls_report[rel_name][score_type])

            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(self.relationNames))),
                                               target_names=self.relationNames)
            print(f'\nTake {take_idx}\n')
            print(cls_report)

        results = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                        target_names=self.relationNames, output_dict=True)
        macro_f1 = results['macro avg']['f1-score']
        self.log(f'Epoch_Proxy_Macro/{split}_PREC', results['macro avg']['precision'])
        self.log(f'Epoch_Proxy_Macro/{split}_REC', results['macro avg']['recall'])
        self.log(f'Epoch_Proxy_Macro/{split}_F1', results['macro avg']['f1-score'])
        self.log(f'Epoch_Proxy_Micro/{split}_PREC', results['weighted avg']['precision'])
        self.log(f'Epoch_Proxy_Micro/{split}_REC', results['weighted avg']['recall'])
        self.log(f'Epoch_Proxy_Micro/{split}_F1', results['weighted avg']['f1-score'])
        print(f'{split} Proxy Results:\n')
        cls_report = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                           target_names=self.relationNames)
        self.logger.experiment.add_text(f'Proxy_Classification_Report/{split}', cls_report, self.current_epoch)
        print(cls_report)
        return macro_f1

    @torch.no_grad()
    def evaluate_with_pred_sg(self, dataset, split):
        dataloader = DataLoader(dataset, shuffle=False, num_workers=self.config['NUM_WORKERS'], pin_memory=True,
                                collate_fn=dataset.collate_fn)  # Create new dataloader
        model_wrapper = SGPNModelWrapper(self.config, num_class=len(dataset.classNames), num_rel=len(dataset.relationNames), weights_obj=dataset.w_cls_obj,
                                         weights_rel=dataset.w_cls_rel, relationNames=dataset.relationNames)  # Create new model
        model_wrapper.load_state_dict(self.state_dict())  # Load weights
        model_wrapper.eval()
        model_wrapper.to(self.device)

        take_rel_preds = defaultdict(list)
        take_rel_gts = defaultdict(list)

        # While running through the dataset, for a take, use the previous frame's predictions as context while making the current prediction.
        model_wrapper.prev_history_per_take = defaultdict(list)
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):  # assumes no shuffle, in order iteration
            # if v has function .to, then v.to(device)
            for k, v in batch.items():
                if hasattr(v, 'to'):
                    batch[k] = v.to(model_wrapper.device)
            if len(model_wrapper.prev_history_per_take[batch['take_idx']]) > 0:
                history = collator.collator([elem['sg'] for elem in model_wrapper.prev_history_per_take[batch['take_idx']]]).to(model_wrapper.device)
            else:
                history = []
            obj_pred, rel_pred, _, _, _, _, probs, _ = model_wrapper(batch, history_sg=history, return_meta_data=True)
            take_rel_preds[batch['take_idx']].extend(rel_pred.detach().argmax(1).cpu().numpy())
            take_rel_gts[batch['take_idx']].extend(batch['gt_rels'].detach().cpu().numpy())

            # if we are using previous frame's predictions as context, then we need to update the history
            predicted_relations = torch.max(rel_pred.detach(), 1)[1]
            relations = []
            for idy, (edge, rel) in enumerate(zip(batch['edge_indices'].transpose(0, 1), predicted_relations)):
                if rel == self.relationNames.index('none'):
                    continue
                start = edge[0]
                end = edge[1]
                start_name = batch['objs_json'][start.item() + 1]
                end_name = batch['objs_json'][end.item() + 1]
                rel_name = self.relationNames[rel]
                relations.append((start_name, rel_name, end_name))

            # map indices to human-readable names both for objects and relationships
            nodes = set()
            for sub, rel, obj in relations:
                nodes.add(sub)
                nodes.add(obj)
                # rel is also a node
                nodes.add(f'$_{rel}')

            nodes = sorted(nodes)
            edges = []
            for sub, rel, obj in relations:
                sub_node = nodes.index(sub)
                obj_node = nodes.index(obj)
                rel_node = nodes.index(f'$_{rel}')
                edges.append((sub_node, rel_node, rel))
                edges.append((rel_node, obj_node, rel))

            data = convert_nodes_edges_to_graph(nodes, edges)
            model_wrapper.prev_history_per_take[batch['take_idx']].append(
                {'scan_id': int(batch['scan_id']), 'sg': collator.preprocess_item(data), 'sg_rels': relations})

        # Starting the actual evaluation
        all_rel_gts = []
        all_rel_preds = []
        for take_idx in sorted(take_rel_preds.keys()):
            rel_preds = take_rel_preds[take_idx]
            rel_gts = take_rel_gts[take_idx]
            all_rel_gts.extend(rel_gts)
            all_rel_preds.extend(rel_preds)
            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(model_wrapper.relationNames))),
                                               target_names=model_wrapper.relationNames, output_dict=True)
            for rel_name in model_wrapper.relationNames:
                for score_type in ['precision', 'recall', 'f1-score']:
                    self.log(f'PredSG_{rel_name}/{take_idx}_{score_type[:2].upper()}', cls_report[rel_name][score_type])

            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(model_wrapper.relationNames))),
                                               target_names=model_wrapper.relationNames)
            print(f'\nTake {take_idx}\n')
            print(cls_report)

        results = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(model_wrapper.relationNames))),
                                        target_names=model_wrapper.relationNames, output_dict=True)

        macro_f1 = results['macro avg']['f1-score']
        self.log(f'Epoch_PredSG_Macro/{split}_PREC', results['macro avg']['precision'])
        self.log(f'Epoch_PredSG_Macro/{split}_REC', results['macro avg']['recall'])
        self.log(f'Epoch_PredSG_Macro/{split}_F1', results['macro avg']['f1-score'])
        self.log(f'Epoch_PredSG_Micro/{split}_PREC', results['weighted avg']['precision'])
        self.log(f'Epoch_PredSG_Micro/{split}_REC', results['weighted avg']['recall'])
        self.log(f'Epoch_PredSG_Micro/{split}_F1', results['weighted avg']['f1-score'])
        print(f'{split} with History Results:\n')
        cls_report = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(model_wrapper.relationNames))),
                                           target_names=model_wrapper.relationNames)
        self.logger.experiment.add_text(f'PredSG_Classification_Report/{split}', cls_report, self.current_epoch)
        print(cls_report)
        model_wrapper.prev_history_per_take = defaultdict(list)
        return macro_f1

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters(), lr=self.lr, weight_decay=float(self.config['W_DECAY']))
        return optimizer
