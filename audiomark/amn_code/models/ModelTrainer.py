"""

"""

import math
import logging

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path
import seaborn as sns

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.nn import functional as F
import torch.utils.data as data_op
from audiomark.amn_code.my_utils import utils
import sklearn.metrics as metrics

logger = logging.getLogger(__name__)


# some common loss functions
def cross_entropy_loss(trainer, data, logits, target, cur_epoch, it, label_smoothing):
    if label_smoothing is None:
        label_smoothing = 0.0

    loss = F.cross_entropy(logits, target, reduction='mean', label_smoothing=label_smoothing)  # sum up batch loss

    return loss


def l2_loss(trainer, data, logits, target, cur_epoch, it, other_data):
    loss = torch.sum((logits-data)**2)
    loss = torch.sqrt(loss) / logits.size(0)

    return loss


class DistillLossCfg(utils.ConfigBase):
    student_alpha = 1.0
    teacher_temp = 6
    teacher_alpha = 150
    teacher_model = None

    teacher_logits_modifier = None


def distill_loss(trainer, data, logits, target, cur_epoch, it, distill_cfg):

    stu_logits = logits

    if np.isclose(distill_cfg.student_alpha, 0.0):
        stu_loss = 0.0  # do not calculate student loss if alpha is 0
    else:
        stu_loss = F.cross_entropy(logits, target, reduction='mean')  # sum up batch loss

    # let the teacher_model to guide training
    with torch.set_grad_enabled(False):
        distill_cfg.teacher_model.eval()     # important to set it to eval
        teacher_logits = distill_cfg.teacher_model(data)

        if distill_cfg.teacher_logits_modifier is not None:
            teacher_logits = distill_cfg.teacher_logits_modifier(trainer, data, logits, target, cur_epoch, it, teacher_logits)

    # from https://github.com/IntelLabs/distiller/blob/master/distiller/knowledge_distillation.py
    # Calculate distillation loss
    soft_log_probs = F.log_softmax(stu_logits / distill_cfg.teacher_temp, dim=1)
    soft_targets = F.softmax(teacher_logits / distill_cfg.teacher_temp, dim=1)

    distillation_loss = F.kl_div(soft_log_probs, soft_targets.detach(), reduction='batchmean')

    # change the original loss
    loss = distill_cfg.student_alpha * stu_loss + distill_cfg.teacher_alpha * distillation_loss

    return loss


class ModelTrainerConfig(utils.ConfigBase):

    # training
    device = torch.device('cuda:0')
    max_epochs = 50
    batch_size = 64
    test_batch_size = 64
    num_workers = 8     # DataLoader
    pin_memory = True   # for Dataloader
    collate_fn = None

    eval_every = None
    test_every = 1

    loss_func = None               # loss_func(trainer, data, logits, target, cur_epoch, it, other_data) will be called when calculating loss
    loss_other_data = None         # passed to loss_func as other_data
    is_classifier = True           # whether training a classifier. if True, will calculate percentage of correct predictions

    # learning
    lr = 1e-3
    lr_gamma = 1.0
    lr_step_size = 50
    grad_norm_clip = 100.0
    weight_decay = 1e-5

    optim_class = optim.AdamW
    optim_kwargs = {}        # besides parameters, lr and weight_decay

    # checkpoint settings
    save_every = None
    ckpt_dir = None

    best_ckpt_dir = None
    best_skip_epochs = 0
    init_best_test_loss = 1e8

    train_epoch_callback = None     # (epoch, train_loss, test_loss, callbk_data). return False -> break the training loop
    train_epoch_callback_data = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.loss_func is None:
            self.loss_func = cross_entropy_loss


class ModelTrainer:
    def __init__(self, model, train_set, test_set, config, valid_set=None):
        self.model = model
        self.config = config

        self.model = self.model.to(self.config.device)

        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.config.batch_size, num_workers=self.config.num_workers,
            pin_memory=config.pin_memory, shuffle=True, collate_fn=config.collate_fn)
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self.config.test_batch_size, num_workers=self.config.num_workers,
            pin_memory=config.pin_memory, shuffle=False, collate_fn=config.collate_fn)

        self.val_loader = None
        if valid_set is not None:
            self.val_loader = torch.utils.data.DataLoader(
                valid_set, batch_size=self.config.test_batch_size, num_workers=self.config.num_workers,
                pin_memory=config.pin_memory, shuffle=False, collate_fn=config.collate_fn)

    def save_ckpt(self, cur_epoch, optimizer, scheduler, train_status_dic, ckpt_dir=None):

        if ckpt_dir is None:
            if self.config.ckpt_dir is None:
                return
            ckpt_dir = Path(self.config.ckpt_dir)

        if not ckpt_dir.exists():
            Path.mkdir(ckpt_dir, parents=True)

        # the dictionary to save
        dic_to_save = {
            "cur_epoch": cur_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_status_dic": train_status_dic,
        }

        save_path = ckpt_dir.joinpath(f"saved_epoch-{cur_epoch}.tar")
        logger.info(f"saving checkpoint: {save_path}", )
        torch.save(dic_to_save, save_path)

        # we also save the loss trends
        sns.set_style("darkgrid")

        plt.Figure()
        plt.plot(list(range(1, cur_epoch+1)), train_status_dic["train_loss_lst"], label="train_loss_lst")
        if len(train_status_dic["val_loss_lst"]) > 0:
            plt.plot(train_status_dic["val_epoch_lst"], train_status_dic["val_loss_lst"], label="val_loss_lst")
        if len(train_status_dic["test_loss_lst"]) > 0:
            plt.plot(train_status_dic["test_epoch_lst"], train_status_dic["test_loss_lst"], label="test_loss_lst")

        # plt.ylim(bottom=-0.1, top=3.0)
        plt.legend()
        plt.savefig(ckpt_dir.joinpath("train_status.png"))
        plt.close()

        # and the error trends
        if self.config.is_classifier:
            plt.Figure()
            train_err_arr = (1 - np.array(train_status_dic["train_acc_lst"])) * 100
            plt.plot(list(range(1, cur_epoch+1)), train_err_arr, label="training error")
            if len(train_status_dic["val_acc_lst"]) > 0:
                val_err_arr = (1 - np.array(train_status_dic["val_acc_lst"])) * 100
                plt.plot(train_status_dic["val_epoch_lst"], val_err_arr, label="validation error")
            if len(train_status_dic["test_acc_lst"]) > 0:
                test_err_arr = (1 - np.array(train_status_dic["test_acc_lst"])) * 100
                plt.plot(train_status_dic["test_epoch_lst"], test_err_arr, label="testing error")

            plt.ylim(bottom=-1, top=100)
            plt.legend()
            plt.savefig(ckpt_dir.joinpath("acc_trends.png"))
            plt.close()

    @staticmethod
    def get_latest_ckpt_path(ckpt_dir):
        # read all *.tar files and get the one with the largest epoch
        latest_epoch = -1
        latest_path = None
        for path in ckpt_dir.glob("*.tar"):
            stem = path.stem
            step_pos = stem.rfind("-")
            epoch = int(stem[step_pos+1:])

            if latest_epoch < epoch:
                latest_epoch = epoch
                latest_path = path

        return latest_path, latest_epoch

    @staticmethod
    def get_ckpt_path(ckpt_dir, epoch):
        ckpt_path = None
        for path in ckpt_dir.glob(f"*-{epoch}.tar"):
            ckpt_path = path
            break

        return ckpt_path

    @staticmethod
    def load_ckpt(ckpt_path, epoch_check=None):
        logging.info(f"loading check point from file: {ckpt_path}, epoch_check = {epoch_check}")
        # load model and optimization weights
        dic_saved = torch.load(ckpt_path)

        if epoch_check:
            assert dic_saved["cur_epoch"] == epoch_check

            train_status_dic = dic_saved["train_status_dic"]
            assert len(train_status_dic['train_loss_lst']) == epoch_check
            assert len(train_status_dic["train_acc_lst"]) == epoch_check

            val_epoch_len = len(train_status_dic["val_epoch_lst"])
            assert len(train_status_dic["val_loss_lst"]) == val_epoch_len
            assert len(train_status_dic['val_acc_lst']) == val_epoch_len
            assert len(train_status_dic["val_fscore_lst"]) == val_epoch_len

            test_epoch_len = len(train_status_dic["test_epoch_lst"])
            assert len(train_status_dic["test_loss_lst"]) == test_epoch_len
            assert len(train_status_dic['test_acc_lst']) == test_epoch_len
            assert len(train_status_dic["test_fscore_lst"]) == test_epoch_len

        return dic_saved

    @staticmethod
    def load_latest_ckpt(ckpt_dir, return_path=False):
        ckpt_dir = Path(ckpt_dir)
        last_ckpt_path, epoch_saved = ModelTrainer.get_latest_ckpt_path(ckpt_dir)
        if last_ckpt_path is None:
            return None

        dic_saved = ModelTrainer.load_ckpt(last_ckpt_path, epoch_check=epoch_saved)

        if return_path is True:
            return dic_saved, last_ckpt_path

        return dic_saved

    @staticmethod
    def prob_to_correct(probs, gtruth):
        pred = torch.argmax(probs, dim=1)
        pred_correct = (pred == gtruth)
        pred_correct = pred_correct.cpu().detach().numpy()
        return pred_correct

    @staticmethod
    def eval_on_dset(model, dset):
        trainer = ModelTrainer(
            model, dset, dset,
            config=ModelTrainerConfig()
        )
        return trainer.eval(trainer.test_loader, "test set")

    def eval(self, dloader, dset_name=""):
        """
        evaluate the model via a specific data loader
        """
        config = self.config
        self.model.eval()

        correct_lst = []
        losses = []
        preds_lst = []
        gtruth_lst = []

        pbar = tqdm(enumerate(dloader), total=len(dloader))
        for it, (x, y) in pbar:
            # place data on the correct device.
            if isinstance(x, torch.Tensor):
                x = x.to(config.device)
            if not isinstance(y, list):
                y = y.to(config.device)

            # forward the model
            with torch.no_grad():
                output = self.model(x)
            loss = config.loss_func(self, x, output, y, None, None, self.config.loss_other_data)

            acc_str = ""
            if config.is_classifier:
                # get the correct prediction
                probs = torch.softmax(output, dim=1)
                pred_correct = self.prob_to_correct(probs, y)
                correct_lst = np.concatenate((correct_lst, pred_correct))
                acc_str = f"accuracy {np.mean(correct_lst):.4f};"

                # we need to return the predictions and ground truth
                preds = torch.argmax(probs, dim=1)
                preds_lst.append(preds.cpu().detach().numpy())
                gtruth_lst.append(y.cpu().detach().numpy())

            losses.append(loss.item())

            # report progress
            pbar.set_description(f"evaluating {dset_name}: loss {np.mean(losses):.5f}; {acc_str}")

        if config.is_classifier is False:
            return np.mean(losses), None, None      # only loss is valid for non classifiers

        preds_lst = np.concatenate(preds_lst)
        gtruth_lst = np.concatenate(gtruth_lst)

        # for binary classification, we also calculate f score
        f_score = None
        if np.unique(gtruth_lst).shape[0] == 2:
            f_score = metrics.f1_score(y_true=gtruth_lst, y_pred=preds_lst, average='binary')

        return np.mean(losses), np.mean(correct_lst), f_score

    def train_epoch(self, dloader, optimizer, scheduler, cur_epoch):
        self.model.train()

        config = self.config
        correct_lst = []
        losses = []

        pbar = tqdm(enumerate(dloader), total=len(dloader))
        for it, (x, y) in pbar:
            # place data on the correct device.
            if isinstance(x, torch.Tensor):
                x = x.to(config.device)
            if not isinstance(y, list):
                y = y.to(config.device)

            # forward the model
            output = self.model(x)
            loss = config.loss_func(self, x, output, y, cur_epoch, it, self.config.loss_other_data)

            # get the correct prediction
            acc_str = ""
            if config.is_classifier:
                pred_correct = self.prob_to_correct(torch.softmax(output, dim=1), y)
                correct_lst = np.concatenate((correct_lst, pred_correct))
                acc_str = f"accuracy {np.mean(correct_lst):.5f}"

            # backprop and update the parameters
            optimizer.zero_grad()
            loss.backward()

            if config.grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_norm_clip)

            optimizer.step()

            # report progress
            losses.append(loss.item())
            pbar.set_description(f"training: epoch {cur_epoch} / {self.config.max_epochs}; "
                                 f"iter {it}; lr {scheduler.get_last_lr()[0]:.2e};"
                                 f" loss {np.mean(losses):.5f}; {acc_str};")

        if len(correct_lst) == 0:
            correct_lst.append(0)   # avoid warning of calculating mean on empty list
        return np.mean(losses), np.mean(correct_lst)

    @staticmethod
    def init_train_status_dic():
        return {
            'train_loss_lst': [],
            "train_acc_lst": [],

            "val_loss_lst": [],
            'val_acc_lst': [],
            "val_fscore_lst": [],  # fscore for binary classification
            "val_epoch_lst": [],

            "test_loss_lst": [],
            'test_acc_lst': [],
            "test_fscore_lst": [],  # fscore for binary classification
            "test_epoch_lst": [],
        }

    def continue_saved_training(self, optimizer, scheduler):
        dic_saved = self.load_latest_ckpt(self.config.ckpt_dir)
        if dic_saved is None:
            return self.init_train_status_dic()

        train_status_dic = dic_saved["train_status_dic"]

        self.model.load_state_dict(dic_saved["model_state"])
        optimizer.load_state_dict(dic_saved["optimizer_state"])
        scheduler.load_state_dict(dic_saved["scheduler_state"])

        logger.info(f"loaded lr = {scheduler.get_last_lr()[0]:e}; "
                    f"scheduler last_epoch = {scheduler.last_epoch}")

        return train_status_dic

    def train(self):
        # optimizer and loader
        optimizer = self.config.optim_class(params=self.model.parameters(), lr=self.config.lr,
                                            weight_decay=self.config.weight_decay, **self.config.optim_kwargs)

        if isinstance(self.config.lr_step_size, list):
            scheduler = MultiStepLR(optimizer, milestones=self.config.lr_step_size, gamma=self.config.lr_gamma)
        else:
            scheduler = StepLR(optimizer, step_size=self.config.lr_step_size, gamma=self.config.lr_gamma)

        # continue saved training
        if self.config.ckpt_dir:
            train_status_dic = self.continue_saved_training(optimizer, scheduler)
            starting_epoch = len(train_status_dic["train_loss_lst"]) + 1
        else:
            # new training
            starting_epoch = 1
            train_status_dic = self.init_train_status_dic()

        best_test_loss = -1e8
        best_ckpt_dir = self.config.best_ckpt_dir
        if best_ckpt_dir is not None:
            best_test_loss = self.config.init_best_test_loss

        for epoch in range(starting_epoch, self.config.max_epochs+1):
            # run one epoch and save print results
            train_loss, train_acc = self.train_epoch(self.train_loader, optimizer, scheduler, epoch)
            scheduler.step()

            train_status_dic["train_loss_lst"].append(train_loss)
            train_status_dic["train_acc_lst"].append(train_acc)

            # may run on the validation set
            if (self.config.eval_every is not None) and (epoch % self.config.eval_every == 0):
                # print(f"evaluating on the validation set:", flush=True)
                val_loss, val_acc, val_fscore = self.eval(self.val_loader)
                train_status_dic["val_loss_lst"].append(val_loss)
                train_status_dic["val_acc_lst"].append(val_acc)
                train_status_dic["val_fscore_lst"].append(val_fscore)
                train_status_dic["val_epoch_lst"].append(epoch)

            # may test
            test_loss = None    # this value may be used in callback
            if (self.config.test_every is not None) and (epoch % self.config.test_every == 0):
                # print(f"evaluating on the test set:", flush=True)
                test_loss, test_acc, test_fscore = self.eval(self.test_loader, "testing set")
                train_status_dic["test_loss_lst"].append(test_loss)
                train_status_dic["test_acc_lst"].append(test_acc)
                train_status_dic["test_fscore_lst"].append(test_fscore)
                train_status_dic["test_epoch_lst"].append(epoch)

            # may save the model
            if ((self.config.save_every is not None) and (epoch % self.config.save_every == 0)) or \
                    (epoch == self.config.max_epochs):
                self.save_ckpt(
                    cur_epoch=epoch,
                    optimizer=optimizer, scheduler=scheduler,
                    train_status_dic=train_status_dic
                )

            if (best_ckpt_dir is not None) and (epoch > self.config.best_skip_epochs) and (test_loss < best_test_loss):
                # first remove existing ckpt
                logging.info(f"saving best tested model --- test_loss = {test_loss}; best_test_loss = {best_test_loss}")
                latest_path, latest_epoch = self.get_latest_ckpt_path(best_ckpt_dir)
                if latest_path is not None:
                    Path.unlink(latest_path)

                self.save_ckpt(
                    cur_epoch=epoch,
                    optimizer=optimizer, scheduler=scheduler,
                    train_status_dic=train_status_dic,
                    ckpt_dir=best_ckpt_dir
                )
                best_test_loss = test_loss

            if self.config.train_epoch_callback is not None:
                if self.config.train_epoch_callback(epoch, train_loss, test_loss, self.config.train_epoch_callback_data) is False:
                    break

        # finish all epochs
        return train_status_dic

    def run(self):
        return self.train()




