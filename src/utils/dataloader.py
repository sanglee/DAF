# @Created   : 2022/10/15 21:41 PM
# @Author    : Sungmin Han
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)
    
def make_dataset(directory, class_to_idx, extensions='.pkl'):
    instances = []
    directory = os.path.expanduser(directory)
    def is_valid_file(x):
        return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class PklsFolder(Dataset):
    def __init__(self, root_dir, dataset_name,flatten,randomize):
        classes, class_to_idx = self._find_classes(root_dir)
        samples = make_dataset(root_dir, class_to_idx)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.dataset_name = dataset_name
        self.flatten = flatten
        self.randomize = randomize

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort(key = lambda x : int(x))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            sample = pickle.load(f)
            
        flow_len = 10
        packet_len = 160
        
        if self.randomize == False:
            flow = np.zeros([flow_len, packet_len])
        else:
            flow = np.random.randint(low=0,high=255,size=(flow_len, packet_len))
            
        for i in range(min(flow_len,len(sample))):
            
            flow[i, :len(sample[i])] = np.frombuffer(sample[i][:packet_len], dtype=np.uint8)
            flow[i, 0:12] = np.random.randint(256, size = 12, dtype = np.uint8)
            flow[i, 26:34] = np.random.randint(256, size = 8, dtype = np.uint8)
        
        if self.flatten == False:
            flow = flow.reshape(40,40)
        else:
            flow = flow.reshape(-1)
        
        if self.dataset_name == 'BoT':
            if target == 10:
                label = 0
            else:
                label = 1
        elif self.dataset_name == 'ToN':
            if target == 1:
                label = 0
            else:
                label = 1
        elif self.dataset_name == 'ISCX2012' or self.dataset_name == 'ISCX2017':
            if target == 0:
                label = 0
            else:
                label = 1
        else:
            assert False, 'Theres no such dataset...'
        
        return flow, label

    def __len__(self):
        return len(self.samples)
    
    def data_cnt_per_class(self):
        class_cnt = {label : 0 for label in self.classes}
        for i in range(len(self.targets)):
            class_cnt[str(self.targets[i])] += 1
        return class_cnt

def make_cls_idx(labels,classes,except_list_path = None):
    
    except_list = []
    
    for except_path in  except_list_path:
        except_list += list(np.load(except_path))
    
    class_idx= {x : [] for x in classes}
    for idx, cls in enumerate(labels):
        class_idx[cls].append(idx)
    
    for key in class_idx.keys():
        class_idx[key] = list(set(class_idx[key]) - set(except_list))
    return class_idx


def split_train_val_test(class_idx_list,train_size,val_size,test_size):
    np.random.shuffle(class_idx_list)
    train_idx = class_idx_list[:train_size]
    val_idx = class_idx_list[train_size:train_size+val_size]
    test_idx = class_idx_list[train_size+val_size:train_size+val_size+test_size]  
    return train_idx,val_idx,test_idx