import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report

import torch
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
from PIL import Image
from bisect import bisect_right
import time


class CosineAnnealingLR_with_Restart(_LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. The original pytorch
    implementation only implements the cosine annealing part of SGDR,
    I added my own implementation of the restarts part.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        T_mult (float): Increase T_max by a factor of T_mult
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        model (pytorch model): The model to save.
        out_dir (str): Directory to save snapshots
        take_snapshot (bool): Whether to save snapshots at every restart

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_mult, model, out_dir, take_snapshot, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.Te = self.T_max
        self.eta_min = eta_min
        self.current_epoch = last_epoch

        self.model = model
        self.out_dir = out_dir
        self.take_snapshot = take_snapshot

        self.lr_history = []

        super(CosineAnnealingLR_with_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lrs = [self.eta_min + (base_lr - self.eta_min) *
                   (1 + math.cos(math.pi * self.current_epoch / self.Te)) / 2
                   for base_lr in self.base_lrs]

        self.lr_history.append(new_lrs)
        return new_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        ## restart
        if self.current_epoch == self.Te:
            print("restart at epoch {:03d}".format(self.last_epoch + 1))

            if self.take_snapshot:
                torch.save({
                    'epoch': self.T_max,
                    'state_dict': self.model.state_dict()
                }, self.out_dir + "Weight/" + 'snapshot_e_{:03d}.pth.tar'.format(self.T_max))

            ## reset epochs since the last reset
            self.current_epoch = 0

            ## reset the next goal
            self.Te = int(self.Te * self.T_mult)
            self.T_max = self.T_max + self.Te

def ccc(x,y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

def ccc_loss(x, y):
    sxy = torch.sum((x - torch.mean(x))*(y - torch.mean(y)))/x.size(0)
    # print(sxy, torch.var(x, unbiased=False), torch.var(y, unbiased=False))
    rhoc = 2*sxy / (torch.var(x, unbiased=False) + torch.var(y, unbiased=False) + (torch.mean(x) - torch.mean(y))**2)
    return 1 - rhoc

def metric_acc(y_va, output_va, y_exp, output_exp, y_au, output_au):
    va_output = output_va.detach().numpy()
    va_true = y_va.detach().numpy()
    exp_output = output_exp.detach().numpy()
    exp_true = y_exp.detach().numpy()
    au_output = output_au.detach().numpy()
    au_true = y_au.detach().numpy()

    arousal_pre = va_output[:, 1]
    arousal_gt = va_true[:, 1]
    valence_pre = va_output[:, 0]
    valence_gt = va_true[:, 0]
    CCC_arousal = ccc(arousal_pre, arousal_gt)
    CCC_valence = ccc(valence_pre, valence_gt)
    metric_ccc = (CCC_arousal + CCC_valence) * 0.5

    exp_pred = exp_output.argmax(axis=1)
    f1_exp = f1_score(exp_true, exp_pred, average='macro')

    au_output[au_output > 0.5] = 1
    au_output[au_output <= 0.5] = 0
    f1_au = f1_score(au_true, au_output, average='macro')

    print(metric_ccc, f1_exp, f1_au)
    metric_all = metric_ccc + f1_exp + f1_au

    return metric_all

def plot_lr_find(lr_list, loss_list):
    lr_list, loss_list = lr_list[10:], loss_list[10:]

    begin_index = int(len(loss_list) / 3)
    min_index = np.argmin(loss_list[begin_index:])
    max_val = np.max(loss_list[:min_index])

    plot_loss_list = [x for x in loss_list if x <= max_val]
    plot_lr_list = [lr_list[i] for i, x in enumerate(loss_list) if x <= max_val]

    fig, ax = plt.subplots()
    ax.plot(plot_lr_list[:-1], plot_loss_list[:-1])
    ax.set_xscale('log')

def lr_find(model, data, train_option, device, lr_init=1e-6, beta=0.98):

    opt = train_option['opt_class'](
        filter( lambda p: p.requires_grad, model.parameters() ),
        lr=lr_init, weight_decay=train_option['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 1.1 ** epoch)

    iter_num, avg_loss, best_loss = 0, 0, float('inf')
    lr_list, loss_list = [], []
    
    os.makedirs('tmp', exist_ok=True)
    torch.save(model.state_dict(), 'tmp/lr_find_before.pkl')
    model.train()

    while True:
        iter_num += 1
        scheduler.step()
        cur_lr = opt.param_groups[0]['lr']
        lr_list.append(cur_lr)

        for X_batch, y_batch in data['train']:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            opt.zero_grad()
            out = model(X_batch)
            loss = train_option['criterion'](out, y_batch)
            loss.backward()
            opt.step()

            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            corr_avg_loss = avg_loss / (1 - beta ** iter_num) # 除以一个值，得到修正版的avg_loss
            loss_list.append(corr_avg_loss)
            best_loss = min(corr_avg_loss, best_loss)
            break


        if (iter_num - 1) % 25 == 0:
            print( 'iter_num=%d, cur_lr=%g, loss=%g' % (iter_num - 1, cur_lr, loss_list[-1]) )
        
            
        if iter_num > 10 and corr_avg_loss > 4 * best_loss:
            break

    model.load_state_dict( torch.load('tmp/lr_find_before.pkl') )
    return lr_list, loss_list

def model_predict(model, data_loader, device,epoch,save_dir):
    model.eval()
    y_true, output = [], []
    os.makedirs(save_dir +('/%d' %epoch), exist_ok=True)
    start_index = 0
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(data_loader):
            X_batch = X_batch.to(device)
            out = model(X_batch)
            output.append( out.to('cpu') )
            y_true.append(y_batch)
            X_batch = X_batch.to('cpu')
            mean = torch.Tensor([0.485,0.456,0.406]).view(1,3,1,1)
            std = torch.Tensor([0.229,0.224,0.225]).view(1,3,1,1)
            X_batch = X_batch * std + mean
            X_batch = X_batch.permute(0,2,3,1)
            X_batch = X_batch.numpy()
            X_batch = np.maximum(0, X_batch)
            X_batch = np.minimum(X_batch, 255)
            X_batch = (X_batch*255).astype(np.uint8) 
            for j in range(X_batch.shape[0]):
                X = Image.fromarray(X_batch[j]) 
                X.save(os.path.join(save_dir,('%d' %epoch),('%d.png' %start_index)))
                start_index+=1
    y_true, output = torch.cat(y_true), torch.cat(output)
    return y_true, output

def model_predict1(model, data_loader, device):
    model.eval()
    y_va, output_va, y_exp, output_exp, y_au, output_au = [], [], [], [], [], []
    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        for i, (X_batch, va_batch, exp_batch, au_batch) in enumerate(data_loader):
            X_batch = X_batch.to(device)
            out_va, out_exp, out_au = model(X_batch)
            out_va = torch.tanh(out_va)
            out_au = sigmoid(out_au)
            output_va.append(out_va.to('cpu'))
            y_va.append(va_batch)
            output_exp.append(out_exp.to('cpu'))
            y_exp.append(exp_batch)
            output_au.append(out_au.to('cpu'))
            y_au.append(au_batch)

    y_va, output_va  = torch.cat(y_va), torch.cat(output_va)
    y_exp, output_exp = torch.cat(y_exp), torch.cat(output_exp)
    y_au, output_au = torch.cat(y_au), torch.cat(output_au)
    return y_va, output_va, y_exp, output_exp, y_au, output_au

def model_predict_AB(modelA, modelB, data_loader, device):
    modelA.eval(), modelB.eval()
    y_true, outputA, outputB = [], [], []

    with torch.no_grad():
        for i, (XA_batch, XB_batch, y_batch) in enumerate(data_loader):
            XA_batch, XB_batch = XA_batch.to(device), XB_batch.to(device)

            outA, outB = modelA(XA_batch), modelB(XB_batch)
            outputA.append( outA.to('cpu') )
            outputB.append( outB.to('cpu') )
            y_true.append(y_batch)

    y_true, outputA, outputB = torch.cat(y_true), torch.cat(outputA), torch.cat(outputB)
    return y_true, outputA, outputB

def model_fit(model, lr, max_epoch, data, train_option, device,save_path, print_interval=1000):
    
    opt = train_option['opt_class'](
        filter( lambda p: p.requires_grad, model.parameters() ),
        lr=lr, weight_decay=train_option['weight_decay']
    )
    
    # scheduler = CosineAnnealingLR_with_Restart(opt,
    #                                           T_max=15,
    #                                           T_mult=1,
    #                                           model=model,
    #                                           out_dir='../input/',
    #                                           take_snapshot=False,
    #                                           eta_min=0)
    
    sigmoid = torch.nn.Sigmoid()
    bce_loss = torch.nn.BCELoss()
    best_metric = 0

    for epoch in range(max_epoch):
        t0, t1 = time.time(), time.time()
        cur_lr = opt.param_groups[0]['lr']
        print('Epoch=%d, lr=%g' % (epoch, cur_lr))

        model.train()
        y_va, output_va, y_exp, output_exp, y_au, output_au = [], [], [], [], [], []

        for batch_i, (X_batch, va_batch, exp_batch, au_batch) in enumerate(data['train']):
            X_batch, va_batch, exp_batch, au_batch = X_batch.to(device), va_batch.to(device), exp_batch.to(device), au_batch.to(device)
            opt.zero_grad()
            out_va, out_exp, out_au = model(X_batch)
            out_va = torch.tanh(out_va)
            loss_va = ccc_loss(out_va[:, 0], va_batch[:, 0]) + ccc_loss(out_va[:, 1], va_batch[:, 1])
            loss_exp = train_option['criterion'](out_exp, exp_batch)
            out_au = sigmoid(out_au)
            loss_au = bce_loss(out_au, au_batch)
            loss = loss_va + loss_exp + loss_au
            loss.backward()
            opt.step()

            y_va.append(va_batch.to('cpu'))
            output_va.append(out_va.to('cpu'))
            y_exp.append(exp_batch.to('cpu'))
            output_exp.append(out_exp.to('cpu'))
            y_au.append(au_batch.to('cpu'))
            output_au.append(out_au.to('cpu'))

            if batch_i % print_interval == 0:
                print('\tbatch_i=%d\tloss=%f\tin %.2f s' %(batch_i, loss.item(), time.time() - t1))
                t1 = time.time()
            
        # scheduler.step()
        y_va, output_va = torch.cat(y_va), torch.cat(output_va)
        y_exp, output_exp = torch.cat(y_exp), torch.cat(output_exp)
        y_au, output_au = torch.cat(y_au), torch.cat(output_au)
        loss_va = ccc_loss(output_va[:, 0], y_va[:, 0]).item() + ccc_loss(output_va[:, 1], y_va[:, 1]).item()
        loss_exp = train_option['criterion_cpu'](output_exp, y_exp).item()
        loss_au = bce_loss(output_au, y_au).item()
        loss_train = loss_va + loss_exp + loss_au
        metric_all = metric_acc(y_va, output_va, y_exp, output_exp, y_au, output_au)
        print('Train\tloss=%f\tmetric_all=%f\tin %.6f s' %
               (loss_train, metric_all, time.time() - t0))
                                   
        if 'valid' in data:
            t0 = time.time()
            y_va, output_va, y_exp, output_exp, y_au, output_au = model_predict1(model, data['valid'], device)

            loss_va = ccc_loss(output_va[:, 0], y_va[:, 0]).item() + ccc_loss(output_va[:, 1], y_va[:, 1]).item()
            loss_exp = train_option['criterion_cpu'](output_exp, y_exp).item()
            loss_au = bce_loss(output_au, y_au).item()
            loss_test = loss_va + loss_exp + loss_au
            metric_all = metric_acc(y_va, output_va, y_exp, output_exp, y_au, output_au)

            ending = '\tBetter!' if metric_all > best_metric else ''
            print('Test\tloss=%f\tmetric_all=%f\tin %.2f s%s' %
                   (loss_test, metric_all, time.time() - t0, ending))
            if metric_all > best_metric:
                torch.save(model.state_dict(), save_path)
                best_metric = metric_all

    print('best_metric = %g' % best_metric)
    
def model_fit_cos(model, lr, max_epoch, data, train_option, device,save_path, count, print_interval=1000):
    
    opt = train_option['opt_class'](
        filter( lambda p: p.requires_grad, model.parameters() ),
        lr=lr, weight_decay=train_option['weight_decay']
    )
    
    scheduler = CosineAnnealingLR_with_Restart(opt,
                                              T_max=15,
                                              T_mult=1,
                                              model=model,
                                              out_dir='../input/',
                                              take_snapshot=False,
                                              eta_min=1e-6)
    
    best_acc = -1 # 存放验证集（或测试集）最优的acc

    for epoch in range(max_epoch):

        t0, t1 = time(), time()
        cur_lr = opt.param_groups[0]['lr']
        print( 'Epoch=%d, lr=%g' % (epoch, cur_lr) )

        model.train()
        y_train, output_train = [], []

        for batch_i, (X_batch, y_batch) in enumerate(data['train']):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            opt.zero_grad()
            out = model(X_batch)
            loss = train_option['criterion'](out, y_batch)
            loss.backward()
            opt.step()

            y_train.append( y_batch.to('cpu') )
            output_train.append( out.to('cpu') )

            if batch_i % print_interval == 0:
                print( '\tbatch_i=%d\tloss=%f\tin %.2f s' %(batch_i, loss.item(), time() - t1) )
                t1 = time()
            
        scheduler.step()
        y_train, output_train = torch.cat(y_train), torch.cat(output_train)
        loss_train = train_option['criterion_cpu'](output_train, y_train).item()
        acc_train, _ = metric_acc(output_train, y_train)
        print('Train\tloss=%f\tacc=%f\tin %.2f s' %
               (loss_train, acc_train, time() - t0))

#         if 'valid' in data:
#             t0 = time()
#             y_test, output_test = model_predict1(model, data['test'], device)
#             loss_test = train_option['criterion_cpu'](output_test, y_test).item()
#             acc_test = metric_acc(output_test, y_test)
            
#             ending = '\tBetter!' if 'valid' not in data and acc_test > best_acc else ''
#             print('Test\tloss=%f\tacc=%f\tin %.2f s%s' %
#                    (loss_test, acc_test, time() - t0, ending))
#             if 'valid' not in data:
#                 if acc_test>best_acc:
#                     torch.save(model.state_dict(),'/home/xiabin/ijcai2020_occ/AffectNet/2019-12-17_clear/util/save_model/best_model.pkl')
#                     best_acc = max(acc_test, best_acc)
                    
                    
        if 'test' in data:
            t0 = time()
            y_test, output_test = model_predict1(model, data['test'], device)
            loss_test = train_option['criterion_cpu'](output_test, y_test).item()
            acc_test, y_pred = metric_acc(output_test, y_test)
            print(classification_report(y_test, y_pred))
            
            ending = '\tBetter!' if 'valid' not in data and acc_test > best_acc else ''
            print('Test\tloss=%f\tacc=%f\tin %.2f s%s' %
                   (loss_test, acc_test, time() - t0, ending))
            if 'valid' not in data:
                if acc_test>best_acc:
                    torch.save(model.state_dict(),os.path.join(save_path,'best_model' + str(count) + '.pkl'))
                    best_acc = max(acc_test, best_acc)
                    
        
        print() # end epoch

    print('best_acc = %g' % best_acc)
    return best_acc