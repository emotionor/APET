import torch
import torch.nn as nn
from model.transformer import Transformer
from utils.builder import get_optimizer, get_lr_scheduler
from utils.metrics import MetricsRecorder
import utils.misc as utils
import time
import datetime
from pathlib import Path
import torch.cuda.amp as amp
import numpy as np


class basemodel(nn.Module):
    def __init__(self, logger, **params) -> None:
        super().__init__()
        self.model = {}
        self.sub_model_name = []
        self.params = params
        self.logger = logger
        self.save_best_param = self.params.get("save_best", "MSE")
        self.metric_best = None
        self.constants_len = self.params.get("constants_len", 0)

        self.begin_epoch = 0
        self.metric_best = 1000

        self.gscaler = amp.GradScaler(init_scale=1024, growth_interval=2000)
        
        # self.whether_final_test = self.params.get("final_test", False)
        # self.predict_length = self.params.get("predict_length", 20)

        # load model
        # print(params)
        sub_model = params.get('sub_model', {})
        # print(sub_model)
        for key in sub_model:
            if key == "transformer":
                self.model[key] = Transformer(**sub_model["transformer"])
            else:
                raise NotImplementedError('Invalid model type.')
            self.sub_model_name.append(key)

        # load optimizer and lr_scheduler
        self.optimizer = {}
        self.lr_scheduler = {}
        self.lr_scheduler_by_step = {}

        optimizer = params.get('optimizer', {})
        lr_scheduler = params.get('lr_scheduler', {})
        # print(optimizer)
        # print(lr_scheduler)
        for key in self.sub_model_name:
            if key in optimizer:
                self.optimizer[key] = get_optimizer(self.model[key], optimizer[key])
            if key in lr_scheduler:
                self.lr_scheduler_by_step[key] = lr_scheduler[key].get('by_step', False)
                self.lr_scheduler[key] = get_lr_scheduler(self.optimizer[key], lr_scheduler[key])

        # load metrics
        eval_metrics_list = params.get('metrics_list', [])
        if len(eval_metrics_list) > 0:
            self.eval_metrics = MetricsRecorder(eval_metrics_list)
        else:
            self.eval_metrics = None

        for key in self.model:
            self.model[key].eval()

    def to(self, device):
        self.device = device
        for key in self.model:
            self.model[key].to(device)
        for key in self.optimizer:
            for state in self.optimizer[key].state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def data_preprocess(self, data):
        inp, pos, target, dos_mean, dos_std = data
        mask = (inp==0)
        inp = inp.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        pos = pos.to(self.device, non_blocking=True)
        dos_mean = dos_mean.to(self.device, non_blocking=True)
        dos_std = dos_std.to(self.device, non_blocking=True)
        mask = torch.tensor(mask.clone().detach(), dtype=torch.bool).to(self.device)
        return inp, pos, mask, target, dos_mean, dos_std

    def loss(self, predict, target):

        #norm = torch.norm(target, p=2)

        return torch.mean(abs(predict-target)) #+ torch.mean((predict-target)**2)*0.02
    
        #return torch.mean((predict-target)**2)
        #return nn.functional.kl_div(predict.softmax(dim=-1).log(), target.softmax(dim=-1), reduction='sum')
        #return self.lossfunc(predict, target)

    def train_one_step(self, batch_data, step):
        inp, pos, mask, target,_,_ = self.data_preprocess(batch_data)
        if len(self.model) == 1:
            predict = self.model[list(self.model.keys())[0]](inp, mask, pos)[0].squeeze(-1)
        else:
            raise NotImplementedError('Invalid model type.')

        loss = self.loss(predict, target)
        if len(self.optimizer) == 1:
            self.optimizer[list(self.optimizer.keys())[0]].zero_grad()
            loss.backward()
            self.optimizer[list(self.optimizer.keys())[0]].step()
        else:
            raise NotImplementedError('Invalid model type.')
        
        return {'loss': loss.item()}

    def multi_step_predict(self, batch_data, clim_time_mean_daily, data_std, index, batch_len):
        pass

    def read_gap(self, array):
        bin=0
        if array[62]==0:
            #bin=1
            pass
        i = 63
        try:
            while array[i]==0:
                bin+=1
                i+=1
        except KeyError as err:
            return 4
        return bin*0.063

    def test_one_step(self, batch_data, step=None, dos_normalize=False, save_predict=False):
        inp, pos, mask, target, dos_mean, dos_std = self.data_preprocess(batch_data)
        if len(self.model) == 1:
            predict, attention = self.model[list(self.model.keys())[0]](inp, mask, pos)
        predict = predict.squeeze(-1)

        loss = self.loss(predict, target)

        data_dict = {}
        data_dict['gt'] = target
        data_dict['pred'] = predict
        metrics_loss = self.eval_metrics.evaluate_batch(data_dict)
        metrics_loss.update({'lp_loss': loss.item()})
        if dos_normalize == True:
            predict_n = predict*dos_std+dos_mean
            target_n = target*dos_std+dos_mean
            predict_n[predict_n < 0] = 0
            MAE_ori = torch.mean(torch.abs((predict_n)-target_n))
            MSE_ori = torch.mean(((predict_n)-target_n)**2)
            metrics_loss.update({'MAE_ori':MAE_ori,'MSE_ori':MSE_ori,'lp_loss': loss.item()})
        if save_predict:

            np.savetxt("dosdata/%s.txt"%step, [predict.squeeze(dim=0).cpu().numpy().T, target.squeeze(dim=0).cpu().numpy().T], fmt="%.4f")
            ###ATTENTION
            #print(attention.squeeze(dim=0).cpu().numpy())
            #with open("selfattention_valid.txt", "a+") as f:
            #    np.savetxt(f, attention.squeeze(dim=0).cpu().numpy().T, fmt='%.04f')
            ###ATTENTION

            ###GAP
            #gaps = [[self.read_gap(predict), self.read_gap(target)]]
            #with open("gap_valid.txt", "a+") as f:
            #    np.savetxt(f, gaps, fmt="%.03f")
            ###GAP
            if False:#loss/torch.norm(predict, p=2) <0.012:
                print(step, loss/torch.norm(target, p=2))
                np.savetxt("save%s_%s.txt"%(step, loss/torch.norm(target, p=2)), [np.array(predict.squeeze(dim=0).cpu()), np.array(target.squeeze(dim=0).cpu())], fmt="%.4f")
        return metrics_loss


    def train_one_epoch(self, train_data_loader, epoch, max_epoches):

        for key in self.lr_scheduler:
            if not self.lr_scheduler_by_step[key]:
                self.lr_scheduler[key].step(epoch)


        # test_logger = {}


        end_time = time.time()           
        for key in self.optimizer:              # only train model which has optimizer
            self.model[key].train()

        metric_logger = utils.MetricLogger(delimiter="  ")
        iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
        data_time = utils.SmoothedValue(fmt='{avg:.3f}')
        max_step = len(train_data_loader)

        header = 'Epoch [{epoch}/{max_epoches}][{step}/{max_step}]'
        for step, batch in enumerate(train_data_loader):

            for key in self.lr_scheduler:
                if self.lr_scheduler_by_step[key]:
                    self.lr_scheduler[key].step(epoch*max_step+step)
        
            # record data read time
            data_time.update(time.time() - end_time)
            # train one step
            loss = self.train_one_step(batch, step)

            # record loss and time
            metric_logger.update(**loss)
            iter_time.update(time.time() - end_time)
            end_time = time.time()

            # output to logger
            if (step+1) % 100 == 0 or step+1 == max_step:
                eta_seconds = iter_time.global_avg * (max_step - step - 1 + max_step * (max_epoches-epoch-1))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.logger.info(
                    metric_logger.delimiter.join(
                        [header,
                        "lr: {lr}",
                        "eta: {eta}",
                        "time: {time}",
                        "data: {data}",
                        "memory: {memory:.0f}",
                        "{meters}"
                        ]
                    ).format(
                        epoch=epoch+1, max_epoches=max_epoches, step=step+1, max_step=max_step,
                        lr=self.optimizer[list(self.optimizer.keys())[0]].param_groups[0]["lr"],
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                        meters=str(metric_logger)
                    ))
                # begin_time1 = time.time()
                # print("logger output time:", begin_time1-end_time)

    def load_checkpoint(self, checkpoint_path):
        checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        checkpoint_model = checkpoint_dict['model']
        checkpoint_optimizer = checkpoint_dict['optimizer']
        checkpoint_lr_scheduler = checkpoint_dict['lr_scheduler']
        for key in checkpoint_model:
            self.model[key].load_state_dict(checkpoint_model[key])
        for key in checkpoint_optimizer:
            self.optimizer[key].load_state_dict(checkpoint_optimizer[key])
        for key in checkpoint_lr_scheduler:
            self.lr_scheduler[key].load_state_dict(checkpoint_lr_scheduler[key])
        self.begin_epoch = checkpoint_dict['epoch']
        if 'metric_best' in checkpoint_dict:
            self.metric_best = checkpoint_dict['metric_best']
        if 'amp_scaler' in checkpoint_dict:
            self.gscaler.load_state_dict(checkpoint_dict['amp_scaler'])
        self.logger.info("last epoch:{epoch}, metric best:{metric_best}".format(epoch=self.begin_epoch, metric_best=self.metric_best))


    def save_checkpoint(self, epoch, checkpoint_savedir, save_type='save_best'): 
        checkpoint_savedir = Path(checkpoint_savedir)
        checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_best.pth' \
                            if save_type == 'save_best' else 'checkpoint_latest.pth')
        # print(save_type, checkpoint_path)
        if utils.get_world_size() > 1:
            utils.save_on_master(
                {
                    'epoch':            epoch+1,
                    'model':            {key: self.model[key].module.state_dict() for key in self.model},
                    'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                    'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                }, checkpoint_path
            )
        else:
            utils.save_on_master(
                {
                    'epoch':            epoch+1,
                    'model':            {key: self.model[key].state_dict() for key in self.model},
                    'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                    'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                }, checkpoint_path
            )

    def whether_save_best(self, metric_logger):
        metric_now = metric_logger.meters[self.save_best_param].global_avg
        if self.metric_best is None:
            self.metric_best = metric_now
            return True
        if metric_now < self.metric_best:
            self.metric_best = metric_now
            return True
        return False



    def trainer(self, train_data_loader, test_data_loader, valid_data_loader, max_epoches, checkpoint_savedir=None, resume=False):
        for epoch in range(self.begin_epoch, max_epoches):

            train_data_loader.sampler.set_epoch(epoch)
            self.train_one_epoch(train_data_loader, epoch, max_epoches)
            # # update lr_scheduler
            # begin_time = time.time()

            
            # begin_time1 = time.time()
            # print("lrscheduler time:", begin_time1 - begin_time)
            # test model
            #metric_logger = self.test(valid_data_loader, epoch)
            metric_logger = self.test(test_data_loader, epoch)
            

            # begin_time2 = time.time()
            # print("test time:", begin_time2 - begin_time1)

            
            # save model
            if checkpoint_savedir is not None:
                if self.whether_save_best(metric_logger):
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_best')
                if (epoch + 1) % 1 == 0:
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_latest')
            # end_time = time.time()
            # print("save model time", end_time - begin_time2)
        

    @torch.no_grad()
    def test(self, test_data_loader, epoch, save_predict=False):
        metric_logger = utils.MetricLogger(delimiter="  ")
        # set model to eval
        for key in self.model:
            self.model[key].eval()


        for step, batch in enumerate(test_data_loader):
            loss = self.test_one_step(batch, save_predict=save_predict, step=step)
            metric_logger.update(**loss)
        
        self.logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))

        return metric_logger

    @torch.no_grad()
    def test_final(self, valid_data_loader, predict_length):
        metric_logger = []
        for i in range(predict_length):
            metric_logger.append(utils.MetricLogger(delimiter="  "))
        # set model to eval
        for key in self.model:
            self.model[key].eval()

        data_mean, data_std = valid_data_loader.dataset.get_meanstd()
        clim_time_mean_daily = valid_data_loader.dataset.get_clim_daily()
        clim_time_mean_daily = clim_time_mean_daily.to(self.device)
        data_std = data_std.to(self.device)
        index = 0
        for step, batch in enumerate(valid_data_loader):
            #print(step)
            batch_len = batch[0].shape[0]
            losses = self.multi_step_predict(batch, clim_time_mean_daily, data_std, index, batch_len)
            for i in range(len(losses)):
                metric_logger[i].update(**losses[i])
            index += batch_len

            self.logger.info("#"*80)

            for i in range(predict_length):
                self.logger.info('  '.join(
                        [f'final valid {i}th step predict (val stats)',
                        "{meters}"]).format(
                            meters=str(metric_logger[i])
                        ))

        return None

    def stat(self):
        pass


