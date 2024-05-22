from __future__ import absolute_import, division, print_function
import os
from models.RexNetV1 import ReXNetV1
from models import BitModel
from Utils import utils as util
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from Utils.dataloader import ImageDataset
import json
import numpy as np
from Utils.metrics import Evaluator
import time, tqdm
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy,SoftTargetCrossEntropy

class Trainer:
    def __init__(self, settings):
        self.settings = settings
        print(self.settings)
        util.init_distributed_mode(self.settings)
        self.device = torch.device(self.settings.device)
        
        # Fix the seed for reproducibility
        seed = util.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True
    
        num_tasks = util.get_world_size()
        global_rank = util.get_rank()
    
        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)
        train_dataset = ImageDataset(self.settings, self.settings.train_csv, self.settings.data_path)
        val_dataset = ImageDataset(self.settings, self.settings.val_csv, self.settings.data_path, train =True)
        test_dataset = ImageDataset(self.settings, self.settings.test_csv, self.settings.data_path, train =True)
        
        sampler_train = torch.utils.data.DistributedSampler(
                train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        self.train_loader = DataLoader(train_dataset, self.settings.batch_size,  sampler= sampler_train, 
                                       num_workers=self.settings.num_workers, pin_memory=self.settings.pin_mem, drop_last=True)
        # torch.autograd.set_detect_anomaly(True)
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.settings.batch_size * self.settings.num_epochs
        
        sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, self.settings.batch_size,  sampler= sampler_val,
                                     num_workers=self.settings.num_workers, pin_memory=self.settings.pin_mem, drop_last=True)
        sampler_test = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        self.test_loader = DataLoader(test_dataset, self.settings.batch_size,  sampler= sampler_test,
                                     num_workers=self.settings.num_workers, pin_memory=self.settings.pin_mem, drop_last=True)
        
        self.settings.num_classes = 5
        if self.model_name == "rexnet":
            self.model = ReXNetV1(classes = self.settings.num_classes, width_mult=3.0)
        else:
            bit_variant = 'BiT-M-R50x1' 
            model = bit_models.KNOWN_MODELS[bit_variant](head_size=n_classes, zero_head=True)
            model.load_from(np.load('weights/BiT-M-R50x1.npz'))
        self.model.to(self.device)
        model_without_ddp = self.model
        if self.settings.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.settings.gpu], find_unused_parameters=True)
            model_without_ddp = self.model.module
        self.parameters_to_train = list(self.model.parameters())

        self.optimizer = optim.Adam(self.parameters_to_train,
                                    self.settings.learning_rate)
        
        if self.settings.load_weights_dir is not None:
            self.load_model()

        ## Print Parameters 
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")
        print("Training model named:\n ", self.settings.model_name)
        print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
        print("Training is using:\n ", self.device)
        self.evaluator = Evaluator(self.settings, self.settings.num_classes)
        self.save_settings()
        self.criterion = torch.nn.CrossEntropyLoss()
        # if mixup_fn is not None:
        #     # smoothing is handled with mixup label transform
        #     criterion = SoftTargetCrossEntropy()
        # elif args.smoothing > 0.:
        #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        # else:
        #     criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        bestv_loss = 100
        v_loss = self.validate()
        for self.epoch in range(self.settings.start_epoch, self.settings.num_epochs):
            self.train_one_epoch()
            v_loss = self.validate()
            if v_loss<bestv_loss:
                bestv_loss = v_loss
                self.save_model()
                print("Best weights are saved")
        print("\n test results\n")
        self.validate_test()
        
    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()

        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))
        m_w = 0
        for batch_idx, (img, one_hot_label, label) in enumerate(pbar):

            outputs, losses = self.process_batch(img, one_hot_label, True)
            self.optimizer.zero_grad(),
            losses.backward()
            self.optimizer.step()
        
    def process_batch(self, img, label, is_training = True):
    
        losses = {}
        img, label = img.to(self.device), label.to(self.device)
    
        outputs = self.model(img)
        loss = self.criterion(outputs, label)
        return outputs, loss

    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("Validating Epoch_{}".format(self.epoch))
        counter = 0
        v_loss = 0
        with torch.no_grad():
            for batch_idx, (img, one_hot_label, label) in enumerate(pbar):
                outputs, losses = self.process_batch(img, label, False)
                v_loss +=losses
                pbar.set_postfix({'TL': v_loss.item()/(1+batch_idx)}, refresh=True)
                self.evaluator.compute_eval_metrics(label,one_hot_label, outputs)
        self.evaluator.print()
        v_loss /=len(self.val_loader)
        del outputs, losses
        return v_loss

    def validate_test(self):
        """Validate the model on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.test_loader)
        pbar.set_description("Validating Epoch_{}".format(self.epoch))
        counter = 0
        with torch.no_grad():
            for batch_idx, (img, one_hot_label, label) in enumerate(pbar):
                outputs, losses = self.process_batch(img, label, False)
                pbar.set_postfix({'TL': v_loss.item()/(1+batch_idx)}, refresh=True)
                self.evaluator.compute_eval_metrics(label,one_hot_label, outputs)
        self.evaluator.print()
        del outputs, losses

            
    def save_settings(self):
        """Save settings to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.settings.__dict__.copy()

        with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model"))
        to_save = self.model.state_dict()
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model from disk
        """
        self.settings.load_weights_dir = os.path.expanduser(self.settings.load_weights_dir)

        assert os.path.isdir(self.settings.load_weights_dir), \
            "Cannot find folder {}".format(self.settings.load_weights_dir)
        print("loading model from folder {}".format(self.settings.load_weights_dir))

        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path, map_location = "cuda:0")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        del pretrained_dict, model_dict
        # loading adam state
        optimizer_load_path = os.path.join(self.settings.load_weights_dir, "adam1.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
