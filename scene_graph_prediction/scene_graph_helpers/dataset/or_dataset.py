from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
# import torch.utils.data as data
from torchvision import transforms as T

from helpers.configurations import OR_4D_DATA_ROOT_PATH
from helpers.utils import load_cam_infos
from scene_graph_prediction.scene_graph_helpers.dataset.augmentation_utils import apply_data_augmentation_to_object_pcs, \
    apply_data_augmentations_to_relation_pcs
from scene_graph_prediction.scene_graph_helpers.dataset.data_preparation_utils import data_preparation, load_full_image_data
from scene_graph_prediction.scene_graph_helpers.dataset.dataset_utils import load_data, get_weights, get_relationships, load_mesh
from scene_graph_prediction.scene_graph_helpers.model.graphormer import collator
from scene_graph_prediction.scene_graph_helpers.model.model_utils import get_image_model


def objname_to_index(objname):
    obj_name_to_index = {
        'anesthesia_equipment': 1,
        'operating_table': 2,
        'instrument_table': 3,
        'secondary_table': 4,
        'instrument': 5,
        'object': 6,
        'human': 7,

        'assisting': 9,
        'cementing': 10,
        'cleaning': 11,
        'closeto': 12,
        'cutting': 13,
        'drilling': 14,
        'hammering': 15,
        'holding': 16,
        'lyingon': 17,
        'operating': 18,
        'preparing': 19,
        'sawing': 20,
        'suturing': 21,
        'touching': 22,
        '[EMPTY]': 23,
        '[MASK]': 24,

    }
    if 'human' in objname or 'Patient' in objname:  # We don't care about patient human_0 human_1 etc. everything is human (We don't seperate patient here, because voxelpose also won't seperate it)
        objname = 'human'
    elif '$' in objname:
        objname = objname.split('_')[1].lower()

    return obj_name_to_index[objname]


index_to_object_name = {
    9: 'Assisting',
    10: 'Cementing',
    11: 'Cleaning',
    12: 'Closeto',
    13: 'Cutting',
    14: 'Drilling',
    15: 'Hammering',
    16: 'Holding',
    17: 'Lyingon',
    18: 'Pperating',
    19: 'Preparing',
    20: 'Sawing',
    21: 'Suturing',
    22: 'Touching',
}


def convert_nodes_edges_to_graph(nodes, edges):
    if len(edges) == 0:
        # Empty graph representation
        data = Data(x=torch.tensor([objname_to_index('[EMPTY]'), objname_to_index('[EMPTY]')], dtype=torch.long).unsqueeze(1),
                    edge_index=torch.tensor([(0, 1)], dtype=torch.long).t().contiguous(), edge_attr=torch.tensor([1], dtype=torch.long),
                    edge_labels=['[EMPTY]'])
    else:
        indices = [list((edge[0], edge[1])) for edge in edges]
        node_features = torch.tensor([objname_to_index(objname) for objname in nodes], dtype=torch.long).unsqueeze(1)
        edge_features = torch.tensor([1 for _ in edges], dtype=torch.long)
        edge_index = torch.tensor(indices, dtype=torch.long)
        data = Data(x=node_features, edge_index=edge_index.t().contiguous(), edge_attr=edge_features, edge_labels=[elem[2] for elem in edges])

    return data


class ORDataset(Dataset):
    def __init__(self,
                 config,
                 split='train',
                 shuffle_objs=False,
                 for_eval=False):

        assert split in ['train', 'val', 'test']
        self.split = split
        self.config = config
        self.mconfig = config['dataset']
        name_suffix = self.mconfig['DATASET_SUFFIX']
        if not self.config['USE_GT']:
            name_suffix += '_no_gt'
        if for_eval:
            name_suffix += '_eval'

        self.caching_folder = Path(f'{OR_4D_DATA_ROOT_PATH}/scene_graph_cache{name_suffix}')
        if not self.caching_folder.exists():
            self.caching_folder.mkdir()
        self.take_to_cam_infos = {}
        for take_idx in range(1, 11):
            self.take_to_cam_infos[take_idx] = load_cam_infos(OR_4D_DATA_ROOT_PATH / f'export_holistic_take{take_idx}_processed')

        self.root = self.mconfig['root']
        self.scans = []
        self.shuffle_objs = shuffle_objs
        self.sample_in_runtime = False
        self.for_eval = for_eval

        self.classNames, self.relationNames, self.data, self.selected_scans = load_data(self.root, self.config, self.mconfig, self.split, self.for_eval)
        self.w_cls_obj, self.w_cls_rel = get_weights(self.classNames, self.relationNames, self.data, self.selected_scans, for_eval=self.for_eval)

        self.relationship_json, self.objs_json, self.scans = get_relationships(self.data, self.selected_scans, self.classNames)

        assert (len(self.scans) > 0)

        self.cache_data = dict()
        self.take_idx_to_human_name_to_3D_joints = {}
        self.take_idx_to_gt_scene_graphs = defaultdict(list)
        self.take_idx_to_pred_scene_graphs = defaultdict(list)

        self.full_image_transformations = get_image_model(model_config=self.config['MODEL'], only_transforms=True)
        if self.full_image_transformations is not None:
            self.full_image_transformations = self.full_image_transformations[split]

            self.image_transform_pre = T.Compose(self.full_image_transformations.transforms[:2])
            self.image_transform_post = T.Compose(self.full_image_transformations.transforms[2:])

        self.empty_sg_raw = Data(x=torch.tensor([objname_to_index('[MASK]'), objname_to_index('[MASK]')], dtype=torch.long).unsqueeze(1),
                                 edge_index=torch.tensor([(0, 1)], dtype=torch.long).t().contiguous(),
                                 edge_attr=torch.tensor([1], dtype=torch.long),
                                 edge_labels=['[MASK]'])
        self.empty_sg = collator.preprocess_item(self.empty_sg_raw)

        if self.config['USE_GT_SGs']:
            for scan in self.scans:
                scan_take_idx, scan_id, _ = scan.split('_')
                rels = self.relationship_json[scan]
                # map indices to human-readable names both for objects and relationships
                nodes = set()
                for sub, obj, _, rel in rels:
                    nodes.add(self.objs_json[scan][sub])
                    nodes.add(self.objs_json[scan][obj])
                    # rel is also a node
                    nodes.add(f'$_{rel}')

                nodes = sorted(nodes)
                edges = []
                for sub, obj, _, rel in rels:
                    sub_node = nodes.index(self.objs_json[scan][sub])
                    obj_node = nodes.index(self.objs_json[scan][obj])
                    rel_node = nodes.index(f'$_{rel}')
                    edges.append((sub_node, rel_node, rel))
                    edges.append((rel_node, obj_node, rel))

                data = convert_nodes_edges_to_graph(nodes, edges)

                self.take_idx_to_gt_scene_graphs[int(scan_take_idx)].append(
                    {'scan_id': int(scan_id), 'sg_raw': data, 'sg': collator.preprocess_item(data), 'sg_rels': rels})

            # make sure every take is correctly sorted by scan id
            for take_idx in self.take_idx_to_gt_scene_graphs:
                self.take_idx_to_gt_scene_graphs[take_idx] = sorted(self.take_idx_to_gt_scene_graphs[take_idx], key=lambda x: x['scan_id'])

    def collate_fn(self, batch):
        batch = batch[0]
        if 'obj_points' in batch:
            batch['obj_points'] = batch['obj_points'].permute(0, 2, 1)
        if 'rel_points' in batch:
            batch['rel_points'] = batch['rel_points'].permute(0, 2, 1)

        batch['gt_class'] = batch['gt_class'].flatten().long()
        batch['edge_indices'] = batch['edge_indices'].t().contiguous()
        batch['take_idx'] = int(batch['scan_id'].split('_')[0])
        if self.config['USE_GT_SGs']:
            predicate_order = {'Cementing': 0, 'Cleaning': 1, 'Cutting': 0, 'Drilling': 0, 'Hammering': 0, 'Operating': 1, 'Preparing': 0,
                               'Sawing': 0, 'Suturing': 0, 'Touching': 2}
            rels_per_scan = [elem['sg_raw'].x.unique().tolist() for elem in batch['gt_sgs_raw']]
            del batch['gt_sgs_raw']
            rel_names_per_scan = [[index_to_object_name[rel] for rel in rels if rel in index_to_object_name] for rels in rels_per_scan]
            main_action_per_scan = []
            for rels in rel_names_per_scan:
                main_action = sorted([rel for rel in rels if rel in predicate_order], key=lambda x: predicate_order[x])
                if len(main_action) == 0:
                    main_action_idx = self.relationNames.index('none')
                else:
                    main_action_idx = self.relationNames.index(main_action[0])
                main_action_per_scan.append(main_action_idx)
            batch['main_actions'] = torch.tensor(main_action_per_scan, dtype=torch.long)

        return batch

    def __len__(self):
        return len(self.scans)

    def _rels_to_hot(self, rels):
        hot_rels = torch.zeros(len(self.relationNames), dtype=torch.long)
        for rel_id in set(rels.unique().cpu().tolist()):
            hot_rels[rel_id] = 1
        return hot_rels

    def __getitem__(self, index):
        scan_id = self.scans[index]
        scan_id_no_split = scan_id.rsplit('_', 1)[0]
        take_idx = scan_id.split('_')[0]
        if take_idx in self.take_idx_to_human_name_to_3D_joints:
            human_name_to_3D_joints = self.take_idx_to_human_name_to_3D_joints[take_idx]
        else:
            human_name_to_3D_joints = np.load(str(OR_4D_DATA_ROOT_PATH / 'human_name_to_3D_joints' / f'{take_idx}_GT_True.npz'), allow_pickle=True)[
                'arr_0'].item()
            self.take_idx_to_human_name_to_3D_joints[take_idx] = human_name_to_3D_joints
        selected_instances = list(self.objs_json[scan_id].keys())
        map_instance2labelName = self.objs_json[scan_id]
        cache_path = self.caching_folder / f'{scan_id}.npz'
        image_input = self.config['IMAGE_INPUT']
        if cache_path.exists():
            sample = np.load(str(cache_path), allow_pickle=True)['arr_0'].item()
        else:
            sample = {'scan_id': scan_id, 'objs_json': self.objs_json[scan_id]}
            data = load_mesh(scan_id_no_split, scan_id, self.objs_json, self.config['USE_GT'], for_infer=self.for_eval,
                             human_name_to_3D_joints=human_name_to_3D_joints)
            points = data['points']
            instances = data['instances']
            instance_label_to_hand_locations = data['instance_label_to_hand_locations']
            obj_points, rel_points, edge_indices, instance2mask, relation_objects_one_hot, gt_rels, gt_class, rel_hand_points = \
                data_preparation(self.config, points, instances, selected_instances, self.mconfig['num_points_objects'],
                                 self.mconfig['num_points_relation'], for_train=True, instance2labelName=map_instance2labelName, classNames=self.classNames,
                                 rel_json=self.relationship_json[scan_id], relationships=self.relationNames, padding=0.2, shuffle_objs=self.shuffle_objs,
                                 instance_label_to_hand_locations=instance_label_to_hand_locations)

            sample['instance2mask'] = instance2mask
            sample['obj_points'] = obj_points
            sample['rel_points'] = rel_points
            sample['gt_class'] = gt_class
            sample['gt_rels'] = gt_rels
            sample['edge_indices'] = edge_indices
            sample['relation_objects_one_hot'] = relation_objects_one_hot
            sample['rel_hand_points'] = rel_hand_points

            np.savez_compressed(str(cache_path), sample)

        if self.split == 'train' and not self.for_eval and self.mconfig['data_augmentation']:
            p_value = 0.75
            if np.random.uniform(0, 1) < p_value:
                sample['obj_points'] = apply_data_augmentation_to_object_pcs(sample['obj_points'])
                sample['rel_points'] = apply_data_augmentations_to_relation_pcs(sample['rel_points'], sample['rel_hand_points'], sample['gt_rels'],
                                                                                self.relationNames)
        if image_input == 'full':
            sample['full_image'] = load_full_image_data(scan_id_no_split, image_transform=self.full_image_transformations,
                                                        augmentations=None)

        scan_idx = int(scan_id.split('_')[1])
        if self.config['USE_GT_SGs']:
            sample['gt_sgs_raw'] = deepcopy(self.take_idx_to_gt_scene_graphs[int(take_idx)])
            # Mask current scan sg
            sample['gt_sgs_raw'][scan_idx] = {'scan_id': int(scan_idx),
                                              'sg_raw': self.empty_sg_raw,
                                              'sg': self.empty_sg}
            sample['gt_sgs'] = collator.collator([elem['sg'] for elem in sample['gt_sgs_raw']])

        sample['scan_idx'] = scan_idx
        sample['hot_rels'] = self._rels_to_hot(sample['gt_rels'])
        return sample
