def get_network(opt_net):
    """Instantiate the network with configuration"""

    kind = opt_net.pop('type').lower()
    print(kind)

    # generators
    if kind == 'sr_resnet':
        from . import SRResNet_arch
        net = SRResNet_arch.SRResNet
    elif kind == 'rrdb_net':  # ESRGAN
        from . import RRDBNet_arch
        net = RRDBNet_arch.RRDBNet
    elif kind == 'mrrdb_net':  # Modified ESRGAN
        from . import RRDBNet_arch
        net = RRDBNet_arch.MRRDBNet
    elif kind == 'ppon':
        from . import PPON_arch
        net = PPON_arch.PPON
    elif kind == 'pan_net':
        from . import PAN_arch
        net = PAN_arch.PAN
    elif kind == 'unet_net':
        from . import UNet_arch
        net = UNet_arch.UnetGenerator
    elif kind == 'resnet_net':
        from . import ResNet_arch
        net = ResNet_arch.ResnetGenerator
    elif kind == 'wbcunet_net':
        from . import WBCNet_arch
        net = WBCNet_arch.UnetGeneratorWBC
    elif kind == 'ddpm':
        from .ddpm_modules import diffusion, unet
        model = unet.UNet(
            in_channel=opt_net['in_channel'],
            out_channel=opt_net['unet']['out_channel'],
            inner_channel=opt_net['unet']['inner_channel'],
            channel_mults=opt_net['unet']['channel_multiplier'],
            attn_res=opt_net['unet']['attn_res'],
            res_blocks=opt_net['unet']['res_blocks'],
            dropout=opt_net['unet']['dropout'],
            image_size=opt_net['diffusion']['image_size']
        )
        net = diffusion.GaussianDiffusion(
            model,
            image_size=opt_net['diffusion']['image_size'],
            channels=opt_net['diffusion']['channels'],
            loss_type='l1',    # L1 or L2
            conditional=opt_net['diffusion']['conditional'],
            schedule_opt=opt_net['beta_schedule']['train']
        )
        if opt['phase'] == 'train':
            # init_weights(net, init_type='kaiming', scale=0.1)
            init_weights(net, init_type='orthogonal')
        if opt['gpu_ids'] and opt['distributed']:
            assert torch.cuda.is_available()
            net = nn.DataParallel(net)
        return net
    elif kind == 'sr3':
        from .sr3_modules import diffusion, unet
        model = unet.UNet(
            in_channel=opt_net['in_channel'],
            out_channel=opt_net['out_channel'],
            inner_channel=opt_net['inner_channel'],
            channel_mults=opt_net['channel_multiplier'],
            attn_res=opt_net['attn_res'],
            res_blocks=opt_net['res_blocks'],
            dropout=opt_net['dropout'],
            image_size=opt_net['image_size']
        )
        net = diffusion.GaussianDiffusion(
            model,
            image_size=opt_net['image_size'],
            channels=opt_net['diffusion_channels'],
            loss_type='l1',    # L1 or L2
            conditional=opt_net['diffusion_conditional'],
            #schedule_opt=opt_net['beta_schedule']['train']
        )
        if opt_net['phase'] == 'train':
            # init_weights(net, init_type='kaiming', scale=0.1)
            init_weights(net, init_type='orthogonal')

        return net
    else:
        raise NotImplementedError('Model [{:s}] not recognized'.format(kind))

    net = net(**opt_net)

    return net
