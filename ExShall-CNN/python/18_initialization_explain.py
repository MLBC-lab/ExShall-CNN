"""
Current:
Replace initializer to uniform distribution

Last:
We replaced all Concat layers with "groupby"

Last:
Hyperparameter Optimization

Last:
In this notebook, we will combine Conv2D with
logarithmic and exponential functions to assimilate
different type of kernels.
For "Retina Blood Vessel" dataset
"""

# Kernels
# 
# exp(log(x) + log(y)) = x * y
# exp(log(x) - log(y)) = x / y
# exp(-|x - y|) = 
#     x > y -> exp(-x + y) = exp(-x) * exp(y) = exp(y) / exp(x)
#     x < y -> exp(x - y) = exp(x) * exp(-y) = exp(x) / exp(y)
# exp(log(|x|) + log(|y|)) = |x| * |y|

# imports
#
import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import shutil
import glob
import time
import copy
import dill
import numpy as np
import scipy as sp
import skimage
from skimage import segmentation, io, filters, morphology
import sklearn
from sklearn import ensemble, metrics, svm
import matplotlib.pyplot as plt
import plotly
import plotly.subplots
import plotly.express as px
import torch, torchvision
import optuna
import plotly
# from plotly import make_subplots
import plotly.express as px

# Globals
#
Train_Image_Ipath = '../RetinaBloodVessels/train/image/'
Train_Mask_Ipath = '../RetinaBloodVessels/train/mask/'
Test_Image_Ipath = '../RetinaBloodVessels/test/image/'
Test_Mask_Ipath = '../RetinaBloodVessels/test/mask/'
ODIR = '../output/shallow'
STUDY = f'study_{time.time():.0f}' if len(sys.argv) == 1 else sys.argv[-1]
NTRIAL = 40
NEPOCH = 400
NBATCH = 4
MIDCHANNEL, NPARALLEL = 1, 2
INIT_RANGE = (-2, 2)
LR_RANGE = (1e-4, 1e-1)
NRADIUS = [0, 1, 2, 4, 6, 8, 10, 12]
# NRADIUS = [0, 1]
NROWS, NCOLS = 512, 512
EPSILON = 1e-6
# br = set_trace
br = breakpoint

# functions and classes

def read_images(path, rescale=True):
    images_fnames = sorted(glob.glob(os.path.join(path, '*.png')))
    images = []
    for fn in images_fnames:
        img = io.imread(fn)
        if rescale:
            img = np.float64(img)
            # Min-Max [-1, +1]
            # img = 2 * (img - img.min()) / (img.max() + EPSILON) - 1
            # img = 2*img - 1
            # img = np.float64(img)/img.max()
            # Normalization
            img = (img - img.mean()) / img.std()
        images.append(img)
    images = np.array(images)
    return images

def gray(img):
    gr = img.mean(axis=2)
    gr = (gr - gr.min()) / (gr.max() - gr.min() + EPSILON)
    return gr

def show(img):
    if img.max != 255:
        img = np.float64(img)
        img = np.uint8(255*(img - img.min())/(img.max()-img.min() + EPSILON))
    fig = plt.figure(figsize=(5, 5))
    ax = fig.subplots()
    ax.imshow(img, cmap='gray')
    return True

class Exp(torch.nn.Module):
    def __init__(self, sat_level=20):
        super(Exp, self).__init__()
        self.sat_level = sat_level
    def forward(self, x):
        out = torch.exp(x)
        if out.isnan().any() | out.isinf().any():
            raise ValueError(f'Exp nan: {out.isnan().any()} '
                             f'inf: {out.isinf().any()}')
        return out

class Log(torch.nn.Module):
    def __init__(self):
        super(Log, self).__init__()
    def forward(self, x):
        out = torch.log(x)
        # out = torch.where(x != 0, torch.log(x), torch.log(x+EPSILON))
        if out.isnan().any() | out.isinf().any():
            raise ValueError(f'Exp nan: {out.isnan().any()}'
                             f'inf: {out.isinf().any()}')
        return out

class JaccardLoss(torch.nn.Module):
    def __init__(self, smooth=1, weight=None, size_average=True):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        # (batch, parallel, height, width)
        smooth = self.smooth
        intersection = (inputs * targets).sum(dim=[0, 2, 3])
        total = (inputs + targets).sum(dim=[0, 2, 3])
        union = total - intersection
        jac = (intersection + smooth)/(union + smooth)
        # return 1 - jac
        return -torch.log(jac)

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        # #comment out if your model contains a sigmoid or equivalent
        # activation layer
        # inputs = torch.sigmoid(inputs)
        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        smooth = self.smooth
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() +
                                           targets.sum() + smooth)
        # return 1 - dice
        return -torch.log(dice)

class DiceBCELoss(torch.nn.Module):
    # Dice Binary Cross Entropy Coefficient
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
    def forward(self, inputs, targets, smooth=1):
        # #comment out if your model contains a sigmoid or equivalent
        # activation layer
        # inputs = torch.sigmoid(inputs)
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs, targets = inputs.flatten(), targets.flatten()
        intersection = (inputs * targets).sum()
        dice_loss = \
            1 - (2.*intersection + smooth)/(inputs.sum() +
                                            targets.sum() + smooth)
        BCE = \
            torch.nn.functional.binary_cross_entropy(inputs,
                                                     targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
    
class Train:
    """
    Initialize, train, evaluate, decode
    """
    def __init__(self, model_dir, data, label, nepoch=4, bsize=8,
                 log_fname='log.txt'):
        """
        data: [batch, num_channel, rows, cols]
        label: [batch, num_channel, rows, cols]
        """
        self.model_dir = model_dir
        self.log_fname = log_fname
        self.data = data
        self.label = label
        self.nepoch = nepoch
        self.device = (torch.device("cuda:0")
                       if torch.cuda.is_available()
                       else torch.device('cpu'))
        log = self.model_dir + '\n'
        write_log(log_fname, log, verbose=True)
        log = f'Device: {self.device}\n'
        write_log(log_fname, log, verbose=True)
        # weight = torch.Tensor([label.sum()/label.numel(),
        #                        1-label.sum()/label.numel()])
        # print('Weight: ', weight)
        # self.crit = torch.nn.CrossEntropyLoss(weight=weight.to(self.device))
        # self.crit = DiceLoss(smooth=0.0)
        # self.crit = DiceBCELoss()
        self.crit = JaccardLoss(smooth=0.0)
        self.bsize = bsize
    
    def __call__(self, trial=None):
        ntrial = 0
        if trial != None:
            ntrial = trial.number
            # lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            lr = trial.suggest_float("lr", *LR_RANGE, log=True)
        # Make the model
        model = LocallyNonlinear(
            ichannel = 3, midchannel=MIDCHANNEL, nparallel=NPARALLEL,
            neighbor_radius = NRADIUS,
            dilation        = [1] * len(NRADIUS))
        for par in model.parameters():
            torch.nn.init.uniform_(par, *INIT_RANGE)
        bmodel, bloss, bloss_index = copy.deepcopy(model), float('inf'), 0
        model = model.to(self.device)
        crit = self.crit
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        # optim = torch.optim.SGD(model.parameters(), lr=lr)
        # # learning rate schedular
        # sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optim, factor=0.7, patience=40, threshold=0.005)
        t1 = time.time()
        for epoch in range(self.nepoch):
            # t1 = time.time()
            # loss = self._valid(model, crit)
            # model, optim = self._train(model, crit, optim)
            # loss = self._valid(model, crit)
            try:
                model, optim = self._train(model, crit, optim)
                loss = self._valid(model, crit)
            except ValueError:
                print(f'A ValueError')
                if trial != None:
                    # trial.report(value=loss.min(), step=epoch)
                    raise optuna.TrialPruned()

            # # learning rate schedular
            # sch.step(loss)
            # if lr != optim.param_groups[0]['lr']:
            #     lr = optim.param_groups[0]['lr']
            #     print(f'Learning rate changed to {lr}.')
            if (epoch % 10 == 0) | (epoch == (self.nepoch-1)):
                log = (f'Ep: {epoch+1}, Secs: {time.time() - t1:.0f}, ' +
                       # f'loss: {torch.round(loss, decimals=4)}\n')
                       f'loss: {[round(x, 2) for x in loss.tolist()]}\n')
                write_log(self.log_fname, log, verbose=True)
                t1 = time.time()
            if loss.min() < bloss:
                bmodel = copy.deepcopy(model)
                bloss, bloss_index = loss.min(), loss.argmin().item()
            if trial != None:
                trial.report(value=loss.min(), step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        with open(os.path.join(self.model_dir, f'{ntrial}.pckl'),
                  mode='wb') as f:
            dill.dump({'model': bmodel.to('cpu'),
                       'bloss': bloss,
                       'bloss_index': bloss_index}, f)
        return loss.min()
            
    def _train(self, model, crit, optim):
        model.train()
        backward_coef = None
        for bc in range(1, self.data.shape[0] // self.bsize + 1 +
                        1*(self.data.shape[0] % self.bsize != 0)):
            slc = slice((bc-1)*self.bsize, bc*self.bsize)
            batch, lb = self.data[slc, :, :, :], self.label[slc].float()
            batch, lb = batch.to(self.device), lb.to(self.device)
            optim.zero_grad()
            out = model(batch)
            lb_repeat = lb.unsqueeze(dim=1).repeat(1, out.shape[1], 1, 1)
            loss = crit(out, lb_repeat)
            if backward_coef == None:
                backward_coef = torch.ones(len(loss), device=self.device)
            loss.backward(backward_coef)
            optim.step()
        return model, optim
    
    def _valid(self, model, crit):
        model = model.to(self.device)
        model.eval()
        loss_sum = 0.0
        with torch.no_grad():
            for bc in range(1, self.data.shape[0] // self.bsize + 1 +
                            1*(self.data.shape[0] % self.bsize != 0)):
                slc = slice((bc-1)*self.bsize, bc*self.bsize)
                batch, lb = self.data[slc], self.label[slc]
                batch, lb = batch.to(self.device), lb.to(self.device).float()
                out = model(batch)
                lb_repeat = \
                    lb.unsqueeze(dim=1).repeat(1, out.shape[1], 1, 1)
                loss = crit(out, lb_repeat)
                loss_sum += loss.detach().cpu()
        return loss_sum / bc

    def decode(self, model):
        model = model.to(self.device)
        model.eval()
        decs = []
        with torch.no_grad():
            for bc in range(1, self.data.shape[0] // self.bsize + 1 +
                            1*(self.data.shape[0] % self.bsize != 0)):
                slc = slice((bc-1)*self.bsize, bc*self.bsize)
                batch, lb = self.data[slc], self.label[slc]
                batch, lb = batch.to(self.device), lb.to(self.device).long()
                out = model(batch)
                # out = out.softmax(dim=1)
                decs += out.detach().cpu().tolist()
        decs = np.array(decs)
        return decs

class LocallyNonlinear(torch.nn.Module):
    # Locally Nonlinear Block
    def __init__(self, ichannel=3, midchannel=4, nparallel=8,
                 neighbor_radius=[3], dilation=[1]):
        super().__init__()
        neighbor = [2*nr+1 for nr in neighbor_radius]
        self.neighbor = neighbor
        self.ichannel = ichannel
        self.midchannel = midchannel
        self.nparallel = nparallel
        layers = torch.nn.ModuleList()
        # Original
        #
        for nei, dil in zip(neighbor, dilation):
            # Summation/Subtraction
            # k1 * x1 + k2 * x2
            #
            layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=ichannel*nparallel*midchannel,
                                out_channels=nparallel,
                                kernel_size=nei, stride=1,
                                dilation=dil, padding='same', bias=True,
                                groups=nparallel,
                                dtype=torch.complex64)))
            # Multiplication/Division/Power
            # exp(k3 * log(k1 * x1) + k4 * log(k2 * x2))
            # Exp(Log(summation))
            #
            layers.append(torch.nn.Sequential(
                Log(),
                torch.nn.Conv2d(in_channels=ichannel*nparallel*midchannel,
                                out_channels=nparallel,
                                kernel_size=1, stride=1,
                                dilation=1, padding='same', bias=True,
                                groups=nparallel,
                                dtype=torch.complex64),
                Exp()))
            # Exp(Log(kernel_summation))
            layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=ichannel*nparallel*midchannel,
                                out_channels=ichannel*nparallel,
                                kernel_size=nei, stride=1,
                                dilation=dil, padding='same', bias=True,
                                groups=nparallel,
                                dtype=torch.complex64), 
                Log(), 
                torch.nn.Conv2d(in_channels=ichannel*nparallel,
                                out_channels=nparallel,
                                kernel_size=1, stride=1,
                                dilation=1, padding='same', bias=True,
                                groups=nparallel,
                                dtype=torch.complex64),
                Exp()))
        self.layers = layers
        self.nlayers = len(layers)
        chc = 0
        with torch.no_grad():
            x = torch.randn(4, ichannel*midchannel*nparallel, 5, 5).cfloat()
            for layer in layers:
                try:
                    out = layer(x)
                except:
                    br()
                chc += out.shape[1]
        # self.dropout = torch.nn.Dropout2d(p=drout)
        self.aggregate = torch.nn.Sequential(
            torch.nn.BatchNorm2d(chc),
            torch.nn.Conv2d(in_channels=chc, out_channels=nparallel,
                            kernel_size=1, stride=1, padding='same',
                            bias=True, groups=nparallel,))
                            # dtype=torch.complex64), )
    def forward(self, x, layers=None):
        if layers == None:
            layers = list(range(self.nlayers))
        y = x.cfloat()
        y = y.repeat(1, self.midchannel*self.nparallel, 1, 1)
        comb = [layer(y) if lc in layers else torch.zeros_like(layer(y))
                for lc, layer in enumerate(self.layers)]
        y = torch.concatenate(comb, dim=1)
        interleave_indices = \
            torch.arange(y.shape[1]).reshape(self.nlayers, -1).t().flatten()
        v = y[:, interleave_indices, :, :]
        v = torch.real(v)
        v = self.aggregate(v)
        v = v.squeeze(dim=1).sigmoid()
        return v

def explain_1(mdl, group=0, rng=[0, 1]):
    # Finds the impact of each kernel by ANOVA
    # Impact = layer_variance_on_data * output_sensitivity
    nf = mdl.aggregate[0].num_features
    points = torch.linspace(rng[0], rng[1], 100)
    mdl.eval()
    bn, conv = mdl.aggregate[0], mdl.aggregate[1]
    variances = np.zeros(nf//2)
    with torch.no_grad():
        for layerc in range(nf//2):
            vals = []
            for point in points:
                x = torch.zeros(1, nf, 1, 1)
                x[0, layerc, 0, 0] = point
                # out = conv(x)
                out = conv(bn(x)).squeeze(dim=1)
                vals.append(out.flatten()[group].item())
            variances[layerc] = np.var(vals)
            # print(layerc, np.var(vals))
    return variances
        
def explain(mdl, images, masks, group=0):
    # Finds the impact of each kernel by removing it in the decode step
    nf = mdl.aggregate[0].num_features
    # Dice score
    dice_score = lambda x, y: \
        2.0 * (x * y).sum() / (x.sum() + y.sum() + EPSILON)
    # Make new model and load the best model values into it
    md = LocallyNonlinear(ichannel=3, midchannel=MIDCHANNEL,
                          nparallel=NPARALLEL,
                          neighbor_radius=NRADIUS,
                          dilation=[1]*len(NRADIUS))
    md.load_state_dict(mdl.state_dict())
    device = (torch.device("cuda:0") if torch.cuda.is_available()
              else torch.device('cpu'))
    masks_float = np.float64(masks)
    layer_impact = np.zeros(nf//2)
    with torch.no_grad():
        decs = decode(md, group, images, batch_size=5,
                      device=device, layers=None)
        top_dice = dice_score(masks_float, decs)
        for layerc in range(nf//2):
            included_layers = list(range(md.nlayers))
            included_layers.remove(layerc)
            # included_layers = [layerc]
            decs = decode(md, group, images, batch_size=5,
                          device=device, layers=included_layers)
            dice = dice_score(masks_float, decs)
            layer_impact[layerc] = (top_dice - dice) / top_dice
    return layer_impact

def decode(model, group, data, batch_size, device,
           layers=None, repeat=False):
    model = model.to(device)
    model.eval()
    decs = []
    with torch.no_grad():
        for bc in range(1, data.shape[0] // batch_size + 1 +
                        1*(data.shape[0] % batch_size != 0)):
            slc = slice((bc-1)*batch_size, bc*batch_size)
            batch = data[slc].to(device)
            if repeat:
                batch = batch.to(device).repeat(1, 2, 1, 1).cfloat()
            if layers:
                out = model(batch, layers)
            else:
                out = model(batch)
            out = out[:, group, :, :]
            out = torch.real(out)
            decs += out.detach().cpu().tolist()
    decs = np.array(decs)
    return decs

# def filter_plot(sl, group, path, nsamples = 100):
#     """
#     noise_shape = (nbatch, ich * ngroup, rows, cols)
#     """
#     for lc in sl:
#         seqlayer = sl[lc]
#         # noise = torch.complex(torch.randn(*noise_shape),
#         #                       torch.randn(*noise_shape))
#         wi = 0
#         while not hasattr(seqlayer[wi], 'weight'):
#             wi += 1
#         ich = seqlayer[wi].in_channels
#         row, col = seqlayer[wi].kernel_size
#         groups = seqlayer[wi].groups
#         x = torch.complex(torch.randn(nsamples, ich, row, col),
#                           torch.randn(nsamples, ich, row, col))
#         with torch.no_grad():
#             y = torch.real(seqlayer(x).detach())
#             y = y[:, group, :, :].squeeze(dim=1)
#         br()

def save_layer_image(layer, images, odir, group):
    device = (torch.device("cuda:0") if torch.cuda.is_available()
              else torch.device('cpu'))
    decs = {lc: decode(layer[lc], group, images, 5, device, repeat=True)
            for lc in layer}
    # Min-Max scaling
    images = (images - images.min()) / (images.max() - images.min())
    # decs = {lc: (d - d.min()) / (d.max() - d.min())
    #         for lc, d in zip(decs.keys(), decs.values())}
            
    # save images and filtered version side by side
    # for c, img, fimg in zip(range(images.shape[0]), images, decs):
    #     row, col = decs.shape[1], decs.shape[2]
    #     image = np.zeros((row, 2*col, 3))
    #     image[:, :col, :] = img.swapdims(0, 1).swapdims(1, 2).numpy()
    #     image[:, col:, :] = fimg[:, :, np.newaxis].repeat(3, axis=2)
    #     image = np.uint8(image * 255)
    #     skimage.io.imsave(os.path.join(odir, f'{c+1:02d}.png'), image)

    for imc in range(images.shape[0]):
        fig = plotly.subplots.make_subplots(
            rows=1, cols=len(layer.keys())+1,
            subplot_titles=['Original']+[f'Kernel {c+1}'
                                         for c in layer.keys()],
            horizontal_spacing=0.01)
        img = images[imc, :, :, :].swapdims(0, 1).swapdims(1, 2)
        img = np.uint8(img * 255)
        fig.add_trace(plotly.graph_objects.Image(z=img), 1, 1)
        for lc, li in enumerate(layer.keys()):
            dec = decs[li][imc, :, :]
            dec = (dec - dec.min()) / (dec.max() - dec.min())
            dec = np.uint8(dec * 255)
            dec = dec[:, :, np.newaxis].repeat(3, axis=2)
            # fig.add_trace(
            #     plotly.graph_objects.Heatmap(z=dec, colorscale='gray'),
            #     1, lc+2)
            # fig.update_layout(coloraxis_showscale=False)
            fig.add_trace(
                plotly.graph_objects.Image(z=dec), 1, lc+2)
        fig.update_yaxes(autorange='reversed', scaleanchor='x',
                         constrain='domain')
        fig.update_xaxes(constrain='domain')
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_annotations(font_size=24)
        fig.update_annotations(y=0.0)
        fig.write_html(os.path.join(odir, f'{imc}.html'))
    return True

def write_log(fname, log, verbose=True):
    if verbose:
        print(log, end='')
    with open(fname, mode='a+') as f:
        f.write(log)
    return True

if __name__ == "__main__":
    model_dir = os.path.join(ODIR, STUDY)
    # if os.path.isdir(model_dir):
    #     shutil.rmtree(model_dir)
    # os.makedirs(model_dir, exist_ok=True)
    log_fname = os.path.join(model_dir, 'log_explain.txt')
    log = sys.argv[0] + '\n'
    write_log(log_fname, log, verbose=True)    
    # read all images and masks and store in two matrices
    train_images = read_images(Train_Image_Ipath, rescale=True)
    train_masks = 1 * (read_images(Train_Mask_Ipath, rescale=False) > 0)
    test_images = read_images(Test_Image_Ipath, rescale=True)
    test_masks = 1 * (read_images(Test_Mask_Ipath, rescale=False) > 0)
    # Print some information
    log = f'Init range: {INIT_RANGE}, Learning rate range: {LR_RANGE}\n'
    log += f'ntrial: {NTRIAL}, nepochs: {NEPOCH}, nbatch: {NBATCH}\n'
    log += f'midchannel: {MIDCHANNEL}, nparallel: {NPARALLEL}\n'
    log += f'neighbor_radius: {NRADIUS}\n'
    write_log(log_fname, log, verbose=True)
    # Print defaults
    log = f'NTRIAL: {NTRIAL}\nNEPOCH: {NEPOCH}\nNBATCH: {NBATCH}\n'
    log += f'MIDCHANNEL, NPARALLEL: {MIDCHANNEL}, {NPARALLEL}\n'
    log += f'INIT_RANGE: {INIT_RANGE}, LR_RANGE: {LR_RANGE}\n'
    log += f'NRADIUS: {NRADIUS}\n'
    write_log(log_fname, log, verbose=True)
    # Print images information
    for lb, imgs, masks in zip(['Train', 'Test'],
                               [train_images, test_images],
                               [train_masks, test_masks]):
        log = lb + '\n'
        log += f'{imgs.shape}, {masks.shape}\n'
        log += f'{imgs.min()}, {imgs.max()}, {imgs.mean()}\n'
        log += f'{masks.min()}, {masks.max()}, {masks.mean()}\n'
        write_log(log_fname, log, verbose=True)
    # convert images to tensors
    train_images_tensors = \
        torch.Tensor(train_images).swapdims(2, 3).swapdims(1, 2)
    test_images_tensors = \
        torch.Tensor(test_images).swapdims(2, 3).swapdims(1, 2)

    # load model to explain
    with open(os.path.join(model_dir, '37.pckl'),
                  mode='rb') as f:
        d = dill.load(f)
        # dill.dump({'model': bmodel.to('cpu'),
        #            'bloss': bloss,
        #            'bloss_index': bloss_index}, f)
    model, bloss, bloss_index = d['model'], d['bloss'], d['bloss_index']
    # layer_impact = explain(model, train_images_tensors, train_masks, group=0)
    # # Plot layers' impacts
    # x = np.arange(len(layer_impact)) + 1
    # y = layer_impact
    # fig = px.bar(x=x, y=y, labels=dict(x='Layer', y='Impact of Layer'))
    # fig.update_traces(texttemplate=[str(a) for a in np.round(100*y, 0)],
    #                   textposition='inside')
    # fig.update_layout(xaxis={'tickmode': 'linear'})
    # fig.layout.font = dict(size=20)
    # fig.write_html(os.path.join(model_dir, f'features_impact.html'))
    
    # select top layers
    # sl_index = np.arange(len(layer_impact))[layer_impact >
    #                                         np.quantile(layer_impact, 0.8)]
    sl_index = [0, 6, 8, 14, 18]
    br()
    print(sl_index)
    sl = {sli: copy.deepcopy(model.layers[sli]) for sli in sl_index}
    # path = os.path.join(model_dir, 'filter')
    # filter_plot(sl, group=0, path=path)
    path = os.path.join(model_dir, 'layer')
    save_layer_image(layer=sl, images=train_images_tensors,
                     odir=path, group=0)
    
    # # make an instance of Train
    # train = Train(model_dir, train_images_tensors, torch.Tensor(train_masks),
    #               nepoch=NEPOCH, bsize=NBATCH,
    #               log_fname=log_fname)
    # # start the training
    # t0 = time.time()
    # # model, loss = train.run(lr=1.0e-4)
    # study = optuna.create_study(study_name=STUDY, direction='minimize')
    # study.optimize(train, n_trials=NTRIAL)
    # ntrial = study.best_trial.number
    # with open(os.path.join(model_dir, f'{ntrial}.pckl'),
    #           mode='rb') as f:
    #     data = dill.load(f)
    # model, bloss, bloss_index = (data['model'], data['bloss'],
    #                              data['bloss_index'])
    # log = f'{model}\n'
    # log += f'Best loss is {study.best_value}.\n'
    # log += f'Best parameters are {study.best_params}.\n'
    # log += f'Best saved loss is: {bloss} in index of {bloss_index}.\n'
    # write_log(log_fname, log, verbose=True)
    # # Jaccard, Dice, etc. on Train and Test
    # #
    # trdecs = decode(model, bloss_index,
    #                 train_images_tensors, torch.Tensor(train_masks),
    #                 5, torch.device('cuda:0'))
    # tsdecs = decode(model, bloss_index,
    #                 test_images_tensors, torch.Tensor(test_masks),
    #                 5, torch.device('cuda:0'))
    # # pr = 1.0 * (decs > 1.5*decs.mean()).flatten()
    # for decs, label in zip([trdecs, tsdecs], [train_masks, test_masks]):
    #     pr = 1.0 * (decs > 0.5*decs.mean()).flatten()
    #     tg = label.flatten()
    #     jaccard = sklearn.metrics.jaccard_score(tg, pr)
    #     dice = 2 * jaccard / (jaccard + 1)
    #     recall = sklearn.metrics.recall_score(tg, pr)
    #     precision = sklearn.metrics.precision_score(tg, pr)
    #     f1score = sklearn.metrics.f1_score(tg, pr)
    #     log = (f'Jaccard: {jaccard:.4f}, Dice: {dice:.4f}, ' +
    #            f'Recall: {recall:.4f}, Precision: {precision:.4f}, ' +
    #            f'F1-score: {f1score:.4f}\n')
    #     write_log(log_fname, log, verbose=True)
    # log = f'Finished in {time.time() - t0:.0f} seconds.\n'
    # write_log(log_fname, log, verbose=True)
    
