from data import EEGDataset, split_data
import torch.utils.data as dataloader
from params import args
from model import *
from utils import *
import numpy as np
import torch as t

class Coach:
    def __init__(self, raw_data):
        trn_data, tst_data = split_data(raw_data, 0.1)
        self.trn_data = EEGDataset(trn_data)
        self.tst_data = EEGDataset(tst_data)
        self.trn_loader = dataloader.DataLoader(self.trn_data, batch_size=args.trn_batch, shuffle=True)
        self.tst_loader = dataloader.DataLoader(self.tst_data, batch_size=args.tst_batch, shuffle=False)
        log('Loaded Data (=ﾟωﾟ)ﾉ')
        
    def run(self):
        self.prepare_model()
        for ep in range(args.trn_epoch):
            res = self.train_epoch()
            log(f'Train {ep}/{args.tst_epoch} {res}')
            if ep % args.tst_epoch == 0:
                res = self.test_epoch()
                log(f'Test {ep}/{args.tst_epoch} {res}')
        res = self.test_epoch()
        log(f'Final Test {res}')

    def prepare_model(self):
        self.encoder = Encoder().cuda()
        self.classifier = Classifier().cuda()
        self.opt = t.optim.Adam(
            [{"params": self.encoder.parameters()},
            {"params": self.encoder.parameters()}],
            lr=args.lr, weight_decay=0
        )

    def train_epoch(self):
        ep_loss, ep_loss_main = 0, 0
        trn_loader = self.handler.trn_loader
        steps = trn_loader.dataset.__len__() // args.trn_batch
        for i, batch_data in enumerate(trn_loader):
            batch_data = [x.cuda() for x in batch_data]

            mat, label = batch_data

            convolutional_embed = self.encoder(mat)
            # sequential_embed = 
            pred = 

            loss_regu = calc_reg_loss(self.encoder) * args.reg
            loss_main = 0 # todo 
            loss = loss_main + loss_regu

            ep_loss += loss.item()
            ep_loss_main += loss_main.item()
            log(f'Step {i}/{steps}: loss = {loss:.3f}, {loss_main:.3f}, loss_regu = {loss_regu:.3f}', online=True)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        res = dict()
        res['loss'] = ep_loss / steps
        res['loss_main'] = ep_loss_main / steps

        return res

    def test_epoch(self):
        tst_loader = self.handler.tst_loader
        ep_loss = 0
        num = tst_loader.dataset.__len__()
        steps = num // args.tst_batch
        for i, batch_data in enumerate(tst_loader):
            batch_data = [x.cuda() for x in batch_data]
            # infrence
            # calculate loss
            # log blablabla i / steps
            # epoch_loss += loss
        res = dict()
        res['loss'] = ep_loss
        return res

if __name__ == '__main__':

    log('Start \(≧▽≦)/')

    # load data

    coach = Coach()
    coach.run()
