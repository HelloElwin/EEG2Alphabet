from data import EEGDataset, get_datasets
from setproctitle import setproctitle
import torch.utils.data as dataloader
from params import args
from model import *
from utils import *
import numpy as np
import torch as t

t.manual_seed(19260817)
np.random.seed(19260817)

setproctitle('eeg@elwin')

class Coach:
    def __init__(self):
        self.trn_data, self.tst_data = get_datasets()
        self.trn_loader = dataloader.DataLoader(self.trn_data, batch_size=args.trn_batch, shuffle=True)
        self.tst_loader = dataloader.DataLoader(self.tst_data, batch_size=args.tst_batch, shuffle=False)
        log(f'Loaded Data (=ﾟωﾟ)ﾉ Train={self.trn_data.__len__()} Test={self.tst_data.__len__()}')
        self.trn_data.prepare_ssl_data()
        log('Prepared SSL Data (=ﾟωﾟ)ﾉ')
        
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
        ep_loss, ep_loss_main, ep_prec, ep_loss_cont, ep_loss_clus = [0] * 5
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
            loss_cont = calc_contrastive_loss(spatial_embed, sequential_embed) * args.reg_cont # todo this is wrong
            loss = loss_main + loss_regu + loss_cont

            ep_loss += loss.item()
            ep_loss_main += loss_main.item()
            ep_loss_cont += loss_cont.item()
            log(f'Step {i}/{steps}: loss = {loss:.3f}, loss_main = {loss_main:.3f}, loss_regu = {loss_regu:.3f}', online=True)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            pred = (pred == t.max(pred, dim=-1, keepdim=True)[0])
            prec = t.sum(label * pred).item()
            ep_prec += prec

        centric_embeds = [[], []]

        for i in range(26):
            mat = t.squeeze(self.trn_data.get_ssl_data(i).cuda())
            loss_clus = 0 # clustering loss

            embed = self.encoder1(t.swapaxes(mat, -1, -2))
            embed_mean = t.mean(embed, axis=0)
            # loss_clus += t.sum(t.sqrt((embed - embed_mean) ** 2)) * args.reg_clus
            centric_embeds[0].append(embed_mean.reshape(1, -1))

            embed = self.encoder2(mat)
            embed_mean = t.mean(embed, axis=0)
            # loss_clus += t.sum(t.sqrt((embed - embed_mean) ** 2)) * args.reg_clus
            centric_embeds[1].append(embed_mean.reshape(1, -1))

            # ep_loss_clus += loss_clus.item()
            log(f'Step(SSL) {i}/26: loss_clus = {loss_clus:.3f}' + ' ' * 50, online=True)

            # self.opt.zero_grad()
            # loss_clus.backward()
            # self.opt.step()

        centric_embeds[0] = t.cat(centric_embeds[0], axis=0)
        centric_embeds[1] = t.cat(centric_embeds[1], axis=0)
        pos = t.exp(t.sum(centric_embeds[0] * centric_embeds[1], axis=1))
        neg = t.sum(t.exp(centric_embeds[0] @ centric_embeds[1].T), axis=1)
        loss_mean_cont = t.sum(-t.log(pos / (neg + 1e-8) + 1e-8)) * args.reg_mean_cont

        self.opt.zero_grad()
        loss_mean_cont.backward()
        self.opt.step()

        res = dict()
        res['loss'] = ep_loss / steps
        res['loss_main'] = ep_loss_main / steps
        res['loss_cont'] = ep_loss_cont / steps
        # res['loss_clus'] = ep_loss_clus / 26
        res['loss_mean_cont'] = loss_mean_cont.item()
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

            spatial_embed = self.encoder1(t.swapaxes(mat, -1, -2))
            sequential_embed = self.encoder2(mat)
            # final_embed = t.cat([spatial_embed, sequential_embed], axis=-1)
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
