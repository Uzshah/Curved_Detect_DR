import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score, f1_score, matthews_corrcoef

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.vals = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']

class Evaluator(object):
    def __init__(self, settings, num_classes=5):
        self.settings = settings
        self.num_classes = num_classes
        self.metrics = {
            "classification/accuracy": AverageMeter(),
            "classification/precision": AverageMeter(),
            "classification/recall": AverageMeter(),
            "classification/auc": AverageMeter(),
            "classification/kappa": AverageMeter(),
            "classification/f1": AverageMeter(),
            "classification/mcc": AverageMeter()
        }

    def reset_eval_metrics(self):
        """Resets metrics used to evaluate the model"""
        for key in self.metrics:
            self.metrics[key].reset()

    def compute_eval_metrics(self, gt_labels, gt_one_hot, pred_probs):
        """
        Computes metrics used to evaluate the model
        Args:
            gt_labels (torch.Tensor): Ground truth labels
            gt_one_hot (torch.Tensor): Ground truth labels (one-hot encoded)
            pred_probs (torch.Tensor): Predicted probabilities (before softmax)
        """
        # Ensure tensors are on the CPU
        pred_probs = F.softmax(pred_probs, dim=1).cpu().numpy()
        pred_labels = np.argmax(pred_probs, axis=1)
        gt_labels = gt_labels.cpu().numpy()
        gt_one_hot = gt_one_hot.cpu().numpy()

        # Flatten arrays for AUC computation
        gt_labels_flat = gt_one_hot.ravel()
        pred_probs_flat = pred_probs.ravel()

        # Compute metrics
        accuracy = (pred_labels == gt_labels).mean()
        precision = np.nanmean([self.precision_per_class(gt_labels, pred_labels, cls) for cls in range(self.num_classes)])
        recall = np.nanmean([self.recall_per_class(gt_labels, pred_labels, cls) for cls in range(self.num_classes)])
        auc = roc_auc_score(gt_labels_flat, pred_probs_flat, multi_class='ovr', average='macro')

        # Check if there are multiple classes in the ground truth labels
        if len(np.unique(gt_labels)) > 1 and len(np.unique(pred_labels)) > 1:
            kappa = cohen_kappa_score(gt_labels, pred_labels, weights='quadratic')
        else:
            kappa = 0.0  # Default to 0 if kappa computation is not possible

        f1 = f1_score(gt_labels, pred_labels, average='macro')
        mcc = matthews_corrcoef(gt_labels, pred_labels)

        # Update metrics
        N = gt_labels.shape[0]
        self.metrics["classification/accuracy"].update(accuracy, N)
        self.metrics["classification/precision"].update(precision, N)
        self.metrics["classification/recall"].update(recall, N)
        self.metrics["classification/auc"].update(auc, N)
        self.metrics["classification/kappa"].update(kappa, N)
        self.metrics["classification/f1"].update(f1, N)
        self.metrics["classification/mcc"].update(mcc, N)

    def precision_per_class(self, gt_labels, pred_labels, cls):
        true_positive = ((pred_labels == cls) & (gt_labels == cls)).sum()
        false_positive = ((pred_labels == cls) & (gt_labels != cls)).sum()
        if true_positive + false_positive == 0:
            return np.nan
        return true_positive / (true_positive + false_positive)

    def recall_per_class(self, gt_labels, pred_labels, cls):
        true_positive = ((pred_labels == cls) & (gt_labels == cls)).sum()
        false_negative = ((pred_labels != cls) & (gt_labels == cls)).sum()
        if true_positive + false_negative == 0:
            return np.nan
        return true_positive / (true_positive + false_negative)

    def print(self, dir=None):
        avg_metrics = []
        avg_metrics.append(self.metrics["classification/accuracy"].avg)
        avg_metrics.append(self.metrics["classification/precision"].avg)
        avg_metrics.append(self.metrics["classification/recall"].avg)
        avg_metrics.append(self.metrics["classification/auc"].avg)
        avg_metrics.append(self.metrics["classification/kappa"].avg)
        avg_metrics.append(self.metrics["classification/f1"].avg)
        avg_metrics.append(self.metrics["classification/mcc"].avg)

        print("\n********************Classification*******************************")
        print("\n  "+ ("{:>12} | " * 7).format("accuracy", "precision", "recall", "auc", "kappa", "f1", "mcc"))
        print(("&  {: 11.5f} " * 7).format(*avg_metrics))

        if dir is not None:
            file = os.path.join(dir, "classification_result.txt")
            with open(file, 'w') as f:
                print("Classification\n  "+ ("{:>12} | " * 7).format("accuracy", "precision", "recall", "auc", "kappa", "f1", "mcc"), file=f)
                print(("&  {: 11.5f} " * 7).format(*avg_metrics), file=f)
