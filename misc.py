from skimage import io
import os
import numpy as np
import torch


# define the loss function (maximum likelihood loss)
def MLloss(y_pred, gt):
    img = y_pred[:,:-1]
    sigma = y_pred[:,-1:] 
    return ( (gt-img)**2 / torch.exp(sigma) + torch.abs(sigma) )/2

# define PSNR
def PSNR(img1, img2, peak=1.0):
    '''
    Computes the PSNR 'metric' between two images assumed to be in the range [0,1]
    '''
    x = (np.array(img1).squeeze() - np.array(img2).squeeze()).flatten() 
    return 10*np.log10(peak**2 / np.mean(x**2))

# save visual results
def save_result(x, y, args, idx):
    print(y.shape)
    b,c,h,w = y.shape
    y_pre = y[:,:c]
    print(y_pre.shape)
    if args.uncertainty:
        sigma = y[:,-1:]

    save_dir = os.path.join(args.save_dir, "images/")
    if not os.path.isdir(save_dir):
        print("Creating dir for saving images.")
        os.makedirs(save_dir)

    for i in range(b):
        if (c-1) == 1:
            img = y_pre[i,0].cpu().numpy()
            n = x[i,0].cpu().numpy()
        else:
            img = y_pre[i].permute(1,2,0).cpu().detach().numpy()
            n = x[i].permute(1,2,0).cpu().numpy()

        io.imsave(save_dir+"{}_denoised.png".format(idx), img)
        io.imsave(save_dir+"{}_noisy.png".format(idx), n)

        if args.uncertainty:    
            uncertainty = sigma[i,0].cpu().numpy()
            io.imsave(save_dir+"{}_sigma.png".format(idx), uncertainty)

def get_capacity(args):
    """
    Usage: arg1=val1-arg2=val2-arg3=val3...
    """

    # load params
    keys = args.capacity.split('-')
    keys = map(lambda x: x.split('='), keys)
    kwargs = { x: y for (x,y) in keys }

    # type conversion
    for k in kwargs.keys():
        if kwargs[k].isnumeric(): # int conversion
            kwargs[k] = int(kwargs[k])

        elif kwargs[k].lower() == "none": # None conversion
            kwargs[k] = None

        else: # bool conversion
            try:
                kwargs[k] = bool(strtobool(kwargs[k]))
            except:
                pass

    return kwargs

def save_options(parser, args):
    """Save all options to a file"""
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    file_name = os.path.join(args.save_dir, 'parser_train.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def training_log(args, epoch, loss, psnr, lr):
    message = 'Epoch {}/{}, loss: {:.4f}, PSNR value: {:.3f}, lr: {:.6f}'.format(epoch, args.num_epochs, loss, psnr, lr)
    print(message)
    logfile_name = os.path.join(args.save_dir, 'loss_log.txt')
    with open(logfile_name, "a") as log_file:
        log_file.write('%s\n' % message)

def message_log(args, message):
    print(message)
    logfile_name = os.path.join(args.save_dir, 'loss_log.txt')
    with open(logfile_name, "a") as log_file:
        log_file.write('%s\n' % message)