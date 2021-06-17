#author: akshitac8
#tf-vaegan inductive
from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import math
import sys
from sklearn import preprocessing
import csv
#import functions
import model
import util
import classifier as classifier
from config import opt
from prototype_loss import compute_prototype_loss
from rw_loss import compute_rw_real_loss, compute_rw_imitative_loss, compute_rw_creative_loss
from sklearn.preprocessing import normalize
# from firelab.config import Config

# rw_config = Config.load(CHANGE_HERE, frozen=False)
# rw_config.freeze()
# print('<=========== Random Walk config ===========>')
# print(rw_config)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# load data
data = util.DATA_LOADER(opt)
# import pdb; pdb.set_trace();
print("# of training samples: ", data.ntrain)

netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
# Init models: Feedback module, auxillary module
netF = model.Feedback(opt)
netDec = model.AttDec(opt,opt.attSize)

print(netE)
print(netG)
print(netD)
print(netF)
print(netDec)

###########
# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize) #attSize class-embedding size
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1
##########
# Cuda
if opt.cuda:
    netD.cuda()
    netE.cuda()
    netF.cuda()
    netG.cuda()
    netDec.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),size_average=False)
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)
           
def sample():
    batch_feature, batch_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)

def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)
    
def generate_syn_feature(generator,classes, attribute,num,netF=None,netDec=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.clone().repeat(num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = Variable(syn_noise,volatile=True)
        syn_attv = Variable(syn_att,volatile=True)
        fake = generator(syn_noisev,c=syn_attv)
        if netF is not None:
            dec_out = netDec(fake) # only to call the forward function of decoder
            dec_hidden_feat = netDec.getLayersOutDet() #no detach layers
            feedback_out = netF(dec_hidden_feat)
            fake = generator(syn_noisev, a1=opt.a2, c=syn_attv, feedback_layers=feedback_out)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label


optimizer          = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerD         = optim.Adam(netD.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG         = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerF         = optim.Adam(netF.parameters(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizerDec       = optim.Adam(netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD,real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

best_gzsl_acc = 0
best_zsl_acc = 0


class FeatDataLayer(object):
    def __init__(self, label, feat_data, attribute, opt):
        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data
        self._label = label
        self._attribute = attribute        
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        # self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""

        if self._cur + self._opt.batch_size >= len(self._label):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self._opt.batch_size]
        self._cur += self._opt.batch_size

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        db_inds = self._get_next_minibatch_inds()
        # import pdb; pdb.set_trace();
        minibatch_feat = np.array([self._feat_data[i].numpy() for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        minibatch_att = np.array([self._attribute[self._label[i]].numpy() for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 'att':minibatch_att}
        return blobs

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label}
        return blobs

# import pdb; pdb.set_trace();
data_layer = FeatDataLayer(data.train_label, data.train_feature, data.attribute, opt)


for epoch in range(0,opt.nepoch):
    for loop in range(0,opt.feedback_loop):
        for i in range(0, data.ntrain, opt.batch_size):
            #########Discriminator training ##############
            for p in netD.parameters(): #unfreeze discrimator
                p.requires_grad = True

            for p in netDec.parameters(): #unfreeze deocder
                p.requires_grad = True
            # Train D1 and Decoder (and Decoder Discriminator)
            gp_sum = 0 #lAMBDA VARIABLE
            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()          
                input_resv = Variable(input_res)
                # print(input_resv.size())
                input_attv = Variable(input_att)

                netDec.zero_grad()
                recons = netDec(input_resv)
                R_cost = opt.recons_weight*WeightedL1(recons, input_attv) 
                R_cost.backward()
                optimizerDec.step()
                # import pdb; pdb.set_trace();
                criticD_real = netD(input_resv, input_attv)
                criticD_real = opt.gammaD*criticD_real.mean()
                criticD_real.backward(mone)
                if opt.encoded_noise:        
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                    eps = Variable(eps.cuda())
                    z = eps * std + means #torch.Size([64, 312])
                else:
                    noise.normal_(0, 1)
                    z = Variable(noise)

                if loop == 1:
                    fake = netG(z, c=input_attv)
                    dec_out = netDec(fake)
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    fake = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                else:
                    fake = netG(z, c=input_attv)

                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = opt.gammaD*criticD_fake.mean()
                criticD_fake.backward(one)
                # gradient penalty
                gradient_penalty = opt.gammaD*calc_gradient_penalty(netD, input_res, fake.data, input_att)
                # if opt.lambda_mult == 1.1:
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()         

                # # Imitative fake RW loss for Discriminator
                # discr_rw_imitative_walker_loss, discr_rw_imitative_visit_loss = compute_rw_imitative_loss(rw_config, data_layer, dataset, netD, netG)
                # discr_rw_imitative_loss = discr_rw_imitative_walker_loss + rw_config.loss_weights.get('visit_loss', 1.0) * discr_rw_imitative_visit_loss
                # discr_rw_imitative_loss = rw_config.loss_weights.discr.imitative * discr_rw_imitative_loss
                # discr_rw_imitative_loss.backward()

                # Imitative real RW loss for Discriminator
                # discr_rw_real_walker_loss, discr_rw_real_visit_loss = compute_rw_real_loss(rw_config, data_layer, dataset, netD, netG)
                # discr_rw_real_loss = discr_rw_real_walker_loss + rw_config.loss_weights.get('visit_loss', 1.0) * discr_rw_real_visit_loss
                # discr_rw_real_loss = rw_config.loss_weights.discr.real * discr_rw_real_loss
                # discr_rw_real_loss.backward()                


                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty #add Y here and #add vae reconstruction loss
                optimizerD.step()

            blobs = data_layer.forward()
            labels = blobs['labels'].astype(int)
            att_creative = Variable(torch.Tensor(blobs['att'].astype(float))).cuda()
            text_feat_1 = np.array([data.train_feature[i, :].numpy() for i in labels])
            text_feat_2 = np.array([data.train_feature[i, :].numpy() for i in labels])
            np.random.shuffle(text_feat_1)  # Shuffle both features to guarantee different permutations
            np.random.shuffle(text_feat_2)
            alpha = (np.random.random(len(labels)) * (.8 - .2)) + .2
            text_feat_mean = np.multiply(alpha, text_feat_1.transpose())
            text_feat_mean += np.multiply(1. - alpha, text_feat_2.transpose())
            text_feat_mean = text_feat_mean.transpose()
            text_feat_mean = normalize(text_feat_mean, norm='l2', axis=1)
            text_feat_Creative = Variable(torch.from_numpy(text_feat_mean.astype('float32'))).cuda()
            z_creative = Variable(torch.randn(opt.batch_size, opt.nz)).cuda()
            # import pdb; pdb.set_trace();
            G_creative_sample = netG(z_creative, c=att_creative)

            gp_sum /= (opt.gammaD*opt.lambda1*opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            #############Generator training ##############
            # Train Generator and Decoder
            for p in netD.parameters(): #freeze discrimator
                p.requires_grad = False
            if opt.recons_weight > 0 and opt.freeze_dec:
                for p in netDec.parameters(): #freeze decoder
                    p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()
            netF.zero_grad()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
            eps = Variable(eps.cuda())
            z = eps * std + means #torch.Size([64, 312])
            if loop == 1:
                recon_x = netG(z, c=input_attv)
                dec_out = netDec(recon_x)
                dec_hidden_feat = netDec.getLayersOutDet()
                feedback_out = netF(dec_hidden_feat)
                recon_x = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
            else:
                recon_x = netG(z, c=input_attv)

            vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var) # minimize E 3 with this setting feedback will update the loss as well
            errG = vae_loss_seen
            
            if opt.encoded_noise:
                criticG_fake = netD(recon_x,input_attv).mean()
                fake = recon_x 
            else:
                noise.normal_(0, 1)
                noisev = Variable(noise)
                if loop == 1:
                    fake = netG(noisev, c=input_attv)
                    dec_out = netDec(recon_x) #Feedback from Decoder encoded output
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    fake = netG(noisev, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                else:
                    fake = netG(noisev, c=input_attv)
                
                criticG_fake = netD(fake,input_attv).mean()

            # # Imitative RW loss for Generator
            # gen_rw_imitative_walker_loss, gen_rw_imitative_visit_loss = compute_rw_imitative_loss(rw_config, data_layer, dataset, netD, netG)
            # gen_rw_imitative_loss = gen_rw_imitative_walker_loss + rw_config.loss_weights.get('visit_loss', 1.0) * gen_rw_imitative_visit_loss
            # gen_rw_imitative_loss = rw_config.loss_weights.gen.imitative * gen_rw_imitative_loss

            # Creative RW loss for Generator
            ## pass netD
            gen_rw_creative_walker_loss, gen_rw_creative_visit_loss = compute_rw_creative_loss(opt, data, netD, G_creative_sample, netG, att_creative=att_creative)
            gen_rw_creative_loss = gen_rw_creative_walker_loss + opt.rw_weight * gen_rw_creative_visit_loss
            gen_rw_creative_loss = opt.creative_weight * gen_rw_creative_loss            
                

            G_cost = -criticG_fake
            errG += opt.gammaG*G_cost
            netDec.zero_grad()
            recons_fake = netDec(fake)
            R_cost = WeightedL1(recons_fake, input_attv)
            errG += opt.recons_weight * R_cost
            errG += gen_rw_creative_loss
            errG.backward()
            # write a condition here
            optimizer.step()
            optimizerG.step()
            if loop == 1:
                optimizerF.step()
            if opt.recons_weight > 0 and not opt.freeze_dec: # not train decoder at feedback time
                optimizerDec.step() 
        
    print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f, gen_creative_loss:%.4f'% (epoch, opt.nepoch, D_cost.data[0], G_cost.data[0], Wasserstein_D.data[0],vae_loss_seen.data[0], gen_rw_creative_loss.data[0]),end=" ")
    netG.eval()
    netDec.eval()
    netF.eval()
    syn_feature, syn_label = generate_syn_feature(netG,data.unseenclasses, data.attribute, opt.syn_num,netF=netF,netDec=netDec)
    # Generalized zero-shot learning
    if opt.gzsl:   
        # Concatenate real seen features with synthesized unseen features
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        # Train GZSL classifier
        gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                25, opt.syn_num, generalized=True, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
        if best_gzsl_acc < gzsl_cls.H:
            best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
        print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H),end=" ")

    # Zero-shot learning
    # Train ZSL classifier
    zsl_cls = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                    data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, \
                    generalized=False, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
    acc = zsl_cls.acc
    if best_zsl_acc < acc:
        best_zsl_acc = acc
    print('ZSL: unseen accuracy=%.4f' % (acc))
    # reset G to training mode
    netG.train()
    netDec.train()
    netF.train()

print('Dataset', opt.dataset)
print('the best ZSL unseen accuracy is', best_zsl_acc)
if opt.gzsl:
    print('Dataset', opt.dataset)
    print('the best GZSL seen accuracy is', best_acc_seen)
    print('the best GZSL unseen accuracy is', best_acc_unseen)
    print('the best GZSL H is', best_gzsl_acc)
