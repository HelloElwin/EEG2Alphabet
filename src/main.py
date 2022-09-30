#encoding:utf-8
import torch.utils.data as dataloader
from params import args
from data import *
from model import *
from utils import *
import numpy as np
import torch as t

t.manual_seed(19260817)
np.random.seed(19260817)

class Coach:
    def __init__(self):
        if args.eval:
            tst_data, tst_label = get_eval_datasets()
            tst_data *= 1000
            self.tst_data = EEGDataset(tst_data, tst_label)
            self.tst_loader = dataloader.DataLoader(self.tst_data, batch_size=args.tst_batch, shuffle=False)
        else:
            trn_data, trn_label, tst_data, tst_label = get_datasets()
            trn_data *= 1000
            tst_data *= 1000
            self.trn_data, self.tst_data = EEGDataset(trn_data, trn_label), EEGDataset(tst_data, tst_label)
            self.trn_loader = dataloader.DataLoader(self.trn_data, batch_size=args.trn_batch, shuffle=True)
            self.tst_loader = dataloader.DataLoader(self.tst_data, batch_size=args.tst_batch, shuffle=False)
            log('Loaded Data (=ﾟωﾟ)ﾉ')

    def save_model(self):
        content = {
            'encoder1': self.encoder1.state_dict(),
            'encoder2': self.encoder2.state_dict(),
            'classifier': self.classifier.state_dict()
        }
        t.save(content, './ckpt/best.ckpt')
        log('Model Saved', bold=True)

    def load_model(self):
        ckp = t.load('./ckpt/best.ckpt')
        self.encoder1.load_state_dict(ckp['encoder1'])
        self.encoder2.load_state_dict(ckp['encoder2'])
        self.classifier.load_state_dict(ckp['classifier'])
        log('Model Loaded!')
        
    def run(self):
        best = 0
        self.prepare_model()
        if args.eval:
            self.load_model()
            res = self.test_epoch()
            log(f"Evaluation accuracy = {res['precision']}", bold=True)
            exit()
        for ep in range(args.trn_epoch):
            res = self.train_epoch()
            log(f'Train {ep}/{args.trn_epoch} {res}')
            if ep % args.tst_epoch == 0:
                res = self.test_epoch()
                log(f'Test {ep}/{args.trn_epoch} {res}')
                if res['precision'] > best:
                    best = res['precision']
                    log(f'Best Result: precision = {best:.4f}', bold=True)
                    self.save_model()
        res = self.test_epoch()
        log(f'Final Test {res}')
        if res['precision'] > best:
            best = res['precision']
            self.save_model()
        log(f'Best Result: precision = {best:.4f}', bold=True)

    def prepare_model(self):
        """
        self.encoders are used to encode representations(embeddings) for each EEG sample
        self.classifier is used to classify the embeddings
        self.contrastive is used for contrastive learning (todo)
        """
        self.encoder1 = SpatialTransformerEncoder().cuda()
        self.encoder2 = TemporalTransformerEncoder().cuda()
        self.classifier = Classifier().cuda()
        # self.contrastive = ContrastiveLearning()
        self.loss_func = nn.CrossEntropyLoss()
        self.opt = t.optim.Adam(
            [{"params": self.encoder1.parameters()},
            {"params": self.encoder2.parameters()},
            {"params": self.classifier.parameters()}],
            lr=args.lr, weight_decay=0
        )

    def train_epoch(self):
        self.encoder1.train()
        self.encoder2.train()
        self.classifier.train()
        ep_loss, ep_loss_main, ep_prec = 0, 0, 0
        trn_loader = self.trn_loader
        steps = trn_loader.dataset.__len__() // args.trn_batch
        for i, batch_data in enumerate(trn_loader):
            batch_data = [x.cuda() for x in batch_data]

            mat, label = batch_data
            mat = t.squeeze(mat) # a batch of EEG samples, (batch_size, num_time_points, num_electrodes)

            spatial_embed = self.encoder1(t.swapaxes(mat, -1, -2))
            sequential_embed = self.encoder2(mat)
            pred = self.classifier(spatial_embed, sequential_embed)

            loss_main = self.loss_func(pred, label) # classification loss
            loss_regu = (calc_reg_loss(self.encoder1) + \
                    calc_reg_loss(self.encoder2) + \
                    calc_reg_loss(self.classifier)) * args.reg # weight regulation loss
            loss_cont = calc_contrastive_loss(spatial_embed, sequential_embed) * args.reg_cont
            loss = loss_main + loss_regu + loss_cont

            ep_loss += loss.item()
            ep_loss_main += loss_main.item()
            log(f'Step {i}/{steps}: loss = {loss:.3f}, loss_main = {loss_main:.3f}, loss_regu = {loss_regu:.3f}', online=True)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            pred = (pred == t.max(pred, dim=-1, keepdim=True)[0])
            prec = t.sum(label * pred).item()
            ep_prec += prec

        res = dict()
        res['loss'] = ep_loss / steps
        res['loss_main'] = ep_loss_main / steps
        res['precision'] = ep_prec / trn_loader.dataset.__len__()

        return res

    def test_epoch(self):
        self.encoder1.eval()
        self.encoder2.eval()
        self.classifier.eval()
        tst_loader = self.tst_loader
        ep_prec = 0
        num = tst_loader.dataset.__len__()
        steps = num // args.tst_batch
        for i, batch_data in enumerate(tst_loader):
            batch_data = [x.cuda() for x in batch_data]
            mat, label = batch_data
            mat = t.squeeze(mat)

            spatial_embed = self.encoder1(t.swapaxes(mat, -1, -2))
            sequential_embed = self.encoder2(mat)
            pred = self.classifier(spatial_embed, sequential_embed)
            pred = (pred == t.max(pred, dim=-1, keepdim=True)[0])

            prec = t.sum(label * pred).item()
            ep_prec += prec
            log(f'Step {i}/{steps}: precision = {prec:.3f}', online=True)
        res = dict()
        res['precision'] = ep_prec / num
        return res

if __name__ == '__main__':

    log('Start \(≧▽≦)/')

    coach = Coach()
    coach.run()
