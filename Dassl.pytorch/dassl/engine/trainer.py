import pdb
import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter,
    AverageMeter,
    tolist_if_not,
    count_num_param,
    load_checkpoint,
    save_checkpoint,
    mkdir_if_missing,
    resume_from_checkpoint,
    load_pretrained_weights,
)
import sys
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator

from collections import defaultdict
import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.spatial.distance import cosine, cdist


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError("Cannot assign model before super().__init__() call")

        if self.__dict__.get("_optims") is None:
            raise AttributeError("Cannot assign optim before super().__init__() call")

        if self.__dict__.get("_scheds") is None:
            raise AttributeError("Cannot assign sched before super().__init__() call")

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name], self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            # pass
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            if self.epoch >= self.max_epoch:
                break
        self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)


class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg, is_dassl=True):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        if is_dassl:
            self.max_epoch = cfg.OPTIM.MAX_EPOCH
            self.output_dir = cfg.OUTPUT_DIR

            self.cfg = cfg
            self.build_data_loader()
            self.build_model()
            self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
            self.best_result = -np.inf
            self.no_improvement = 0

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)
        if self.cfg.TEST.OOD_DATASET:
            print("Using OOD test dataset")
            dm2 = DataManager(self.cfg, False)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        if self.cfg.TEST.OOD_DATASET:
            print("Using OOD test dataset")
            self.test_loader = dm2.test_loader
            self.all_loader = dm2.all_loader
        else:
            self.test_loader = dm.test_loader
            if dm.all_loader is not None:
                self.all_loader = dm.all_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm
        if self.cfg.TEST.OOD_DATASET:
            self.dm2 = dm2

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        sys.stdout.flush()

        # Close writer
        self.close_writer()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST_EARLY
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0
            else False
        )
        # print("test????????????? {}, {}".format(do_test, self.cfg.TRAIN.PATIENT_COUNT))
        if do_test and self.cfg.TRAIN.PATIENT_COUNT != -1:
            # print("test!!!!!!!!!!!!!!!!!!")
            with torch.no_grad():
                curr_result = self.test(split="val")
                # 加入提前停止，如果验证集上的性能没有提升次数大于3次，则停止训练
                if self.cfg.TRAIN.EARLY_STOPPING:
                    if curr_result > self.best_result:
                        self.no_improvement = 0
                    else:
                        self.no_improvement += 1
                        if (
                            self.no_improvement >= self.cfg.TRAIN.PATIENT_COUNT
                            and self.epoch >= 5
                        ):
                            print(f"Early stopping at epoch {self.epoch}")
                            self.epoch = self.max_epoch + 1
                            return
                print(f"now no improvement {self.no_improvement} times")
                if curr_result > self.best_result + 0.01:
                    self.best_result = curr_result
                    self.save_model(
                        self.epoch,
                        self.output_dir,
                        val_result=curr_result,
                        model_name="model-best.pth.tar",
                    )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input, return_features=False):
        if return_features:
            return self.model(input, return_features=True)
        else:
            return self.model(input)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        # 改为使用get_last_lr

        return self._optims[name].param_groups[0]["lr"]


class TrainerXU(SimpleTrainer):
    """A base trainer using both labeled and unlabeled data.

    In the context of domain adaptation, labeled and unlabeled data
    come from source and target domains respectively.

    When it comes to semi-supervised learning, all data comes from the
    same domain.
    """

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u


class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        epoch_begin = time.time()
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
        # print("total epoch time: {}".format(epoch_begin - end))

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain

    def draw_distribution(self):
        from sklearn.manifold import TSNE

        images = []
        labels = []
        count = {}
        for batch in self.test_loader:
            image, label = self.parse_batch_train(batch)
            for i in range(len(label)):
                # print(label[i].cpu().numpy())
                num = label[i].cpu().numpy().item()
                if num not in range(50, 70, 1):
                    # print(label[i])
                    continue
                if num in count:
                    if count[num] >= 70:
                        continue
                    count[num] += 1
                    images.append(image[i])
                    labels.append(label[i])
                else:
                    count[num] = 1
                    images.append(image[i])
                    labels.append(label[i])

        # print(count.keys())
        # 将图片和标签转化为tensor
        images = torch.stack(images)
        labels = torch.stack(labels)

        image_features, text_features = self.get_features(images)

        # 使用PCA降维
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        tsne = TSNE(n_components=2, random_state=42)
        image_features = tsne.fit_transform(image_features.cpu().numpy())
        # text_features = tsne.fit_transform(text_features.cpu().numpy())
        # 图像特征放大
        # image_features = image_features * 5
        # 将特征归一化到-10到10之间
        image_features = (image_features - image_features.min()) / (
            image_features.max() - image_features.min()
        ) * 10 - 5

        plt.figure(figsize=(5, 5))
        plt.xticks([])
        plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.scatter(
            image_features[:, 0],
            image_features[:, 1],
            c=labels.cpu().numpy(),
            cmap="tab20",
            s=10,
        )
        out_dir = self.cfg.OUTPUT_DIR + "/image_features_{}_{}_trans.png".format(
            self.cfg.TRAINER.NAME, self.cfg.DATASET.NAME
        )
        plt.savefig(out_dir, dpi=300, transparent=True)
        print("picture saved in {}".format(out_dir))

    def cal_distance(self):
        images = []
        labels = []
        count = {}
        for batch in self.test_loader:
            image, label = self.parse_batch_train(batch)
            for i in range(len(label)):
                # print(label[i].cpu().numpy())
                num = label[i].cpu().numpy().item()
                # if num not in range(180, 220, 1):
                #     # print(label[i])
                #     continue
                if num in count:
                    if count[num] >= 10:
                        continue
                    count[num] += 1
                    images.append(image[i])
                    labels.append(label[i])
                else:
                    count[num] = 1
                    images.append(image[i])
                    labels.append(label[i])

        # print(count.keys())
        # 将图片和标签转化为tensor
        images = torch.stack(images)
        labels = torch.stack(labels)

        image_features, text_features = self.get_features(images)

        # # 使用PCA降维
        # from sklearn.manifold import TSNE
        # tsne = TSNE(n_components=2, random_state=9)
        # image_features = tsne.fit_transform(image_features.cpu().numpy())
        # text_features = tsne.fit_transform(text_features.cpu().numpy())

        # 计算每个图像与其标签的距离
        distance = []
        for i in range(len(image_features)):
            distance.append(
                np.linalg.norm(
                    image_features[i].cpu().numpy()
                    - text_features[labels[i]].cpu().numpy()
                )
            )

        print("mean distance: ", np.mean(distance))
        print("std distance: ", np.std(distance))
        print("max distance: ", np.max(distance))
        print("min distance: ", np.min(distance))

        with open(
            "/home/lvsl/Code/BITP/{}_distance.txt".format(self.cfg.TRAINER.NAME), "a"
        ) as f:
            f.write(str(np.mean(distance)) + "\t")
            f.write(str(np.std(distance)) + "\t")
            f.write(str(np.max(distance)) + "\t")
            f.write(str(np.min(distance)) + "\t")
            f.write("\n")

    @staticmethod
    # 对齐聚类标签和真实标签
    def cluster_accuracy(true_labels, cluster_labels):
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        """
        将聚类标签重新排列，使其与真实标签尽可能匹配
        """
        D = max(cluster_labels.max(), true_labels.max()) + 1
        cost_matrix = np.zeros((D, D), dtype=np.int64)

        for i in range(len(cluster_labels)):
            cost_matrix[cluster_labels[i], true_labels[i]] += 1

        row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
        return cost_matrix[row_ind, col_ind].sum() / len(cluster_labels)

    def group_features_by_class(self, features, labels):
        # 创建一个字典，键是标签，值是相应类的特征列表
        grouped_features = defaultdict(list)
        for feature, label in zip(features, labels):
            grouped_features[label].append(feature)
        # 转换列表为数组方便后续计算
        return {
            label: np.array(features) for label, features in grouped_features.items()
        }

    def calculate_intra_class_metrics(self, features, labels):
        grouped_features = self.group_features_by_class(features, labels)
        # print("grouped_features: ", grouped_features.shape)
        metrics = {}

        for label, class_features in grouped_features.items():
            # 确保类特征矩阵的形状正确
            if class_features.shape[0] < 2:
                continue  # 若某类样本数不足，跳过该类

            # 计算各类内的不同相关性指标
            metrics[label] = {
                "intra_class_avg_distance": self.intra_class_average_distance(
                    class_features
                ),
                "intra_class_variance": self.intra_class_variance(class_features),
                "std_dev_euclidean": self.std_dev_euclidean(class_features),
                "mean_similarity_to_centroid": self.mean_similarity_to_centroid(
                    class_features
                ),
            }
        average_dis = []
        average_variance = []
        average_dev = []
        average_sim = []

        for key in metrics:
            for metric in metrics[key]:
                if metric == "intra_class_avg_distance":
                    average_dis.append(metrics[key][metric])
                if metric == "intra_class_variance":
                    average_variance.append(metrics[key][metric])
                if metric == "std_dev_euclidean":
                    average_dev.append(metrics[key][metric])
                if metric == "mean_similarity_to_centroid":
                    average_sim.append(metrics[key][metric])

                # print("{}, {}, {}".format(key, metric, metrics[key][metric]))

        # please print all average value
        print(
            f"|{np.mean(average_dis)}|{np.mean(average_variance)}|{np.mean(average_dev)}|{np.mean(average_sim)}|"
        )

        return metrics

    def intra_class_average_distance(self, features):
        # `features` should be an array of shape (n_samples, n_features) for one class
        return pdist(features, metric="euclidean").mean()

    def intra_class_variance(self, features):
        # features is assumed to be a (n_samples, n_features) matrix
        return np.var(features, axis=0).mean()

    def std_dev_euclidean(self, features):
        distances = pdist(features, metric="euclidean")
        return distances.std()

    def mean_similarity_to_centroid(self, features):
        centroid = features.mean(axis=0)
        distances = cdist(features, [centroid], metric="euclidean")
        return distances.mean()
