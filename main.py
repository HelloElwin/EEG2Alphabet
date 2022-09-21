from data import EEGDataset, get_datasets
import torch.utils.data as dataloader
from params import args
from model import *
from utils import *
import numpy as np
import torch as t

t.manual_seed(19260817)
np.random.seed(19260817)

class Coach:
    def __init__(self):
        self.trn_data, self.tst_data = get_datasets()
        self.trn_loader = dataloader.DataLoader(self.trn_data, batch_size=args.trn_batch, shuffle=True)
        self.tst_loader = dataloader.DataLoader(self.tst_data, batch_size=args.tst_batch, shuffle=False)
        log('Loaded Data (=ﾟωﾟ)ﾉ')
        
    def run(self):
        self.prepare_model()
        for ep in range(args.trn_epoch):
            res = self.train_epoch()
            log(f'Train {ep}/{args.trn_epoch} {res}')
            if ep % args.tst_epoch == 0:
                res = self.test_epoch()
                log(f'Test {ep}/{args.trn_epoch} {res}')
        res = self.test_epoch()
        log(f'Final Test {res}')

    def prepare_model(self):
        """
        self.encoders are used to encode representations(embeddings) for each EEG sample
        self.classifier is used to classify the embeddings
        self.contrastive is used for contrastive learning (todo)
        """
        self.encoder1 = ResNetEncoder().cuda()
        self.encoder2 = TransformerEncoder().cuda()
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
        ep_loss, ep_loss_main, ep_prec = 0, 0, 0
        trn_loader = self.trn_loader
        steps = trn_loader.dataset.__len__() // args.trn_batch
        for i, batch_data in enumerate(trn_loader):
            batch_data = [x.cuda() for x in batch_data]

            mat, label = batch_data
            mat = t.squeeze(mat) # a batch of EEG samples, (batch_size, num_time_points, num_electrodes)

            convolutional_embed = self.encoder1(t.unsqueeze(mat, axis=1))
            sequential_embed = self.encoder2(mat)
            # final_embed = t.cat([convolutional_embed, sequential_embed], axis=-1)
            pred = self.classifier(convolutional_embed, sequential_embed)

            loss_main = self.loss_func(pred, label) # classification loss
            loss_regu = (calc_reg_loss(self.encoder1) + \
                    calc_reg_loss(self.encoder2) + \
                    calc_reg_loss(self.classifier)) * args.reg # weight regulation loss
            # loss_cont = calc_contrastive_loss(convolutional_embed, sequential_embed) * args.cl_reg # todo this is wrong
            loss = loss_main + loss_regu # + loss_cont

            ep_loss += loss.item()
            ep_loss_main += loss_main.item()
            log(f'Step {i}/{steps}: loss = {loss:.3f}, {loss_main:.3f}, loss_regu = {loss_regu:.3f}', online=True)

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
        tst_loader = self.tst_loader
        ep_prec = 0
        num = tst_loader.dataset.__len__()
        steps = num // args.tst_batch
        for i, batch_data in enumerate(tst_loader):
            batch_data = [x.cuda() for x in batch_data]
            mat, label = batch_data
            mat = t.squeeze(mat)

            convolutional_embed = self.encoder1(t.unsqueeze(mat, axis=1))
            sequential_embed = self.encoder2(mat)
            # final_embed = t.cat([convolutional_embed, sequential_embed], axis=-1)
            pred = self.classifier(convolutional_embed, sequential_embed)
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
