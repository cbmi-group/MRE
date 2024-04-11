# This file is used to build the model based on the model type and loss criterion
from models.pvt_MRE import PVTMRE_SingleLoss,PVTMRE_Up_Sampling_Loss,PVTMRE_Multi_Layer_Loss,PVTMRE_Hierarchical_Fusing_Loss
from models.unet_MRE import UNetMRE_Single_Loss,UNetMRE_Up_Sampling_Loss,UNetMRE_Multi_Layer_Loss,UNetMRE_Hierarchical_Fusing_Loss


def build_model(opts):
    model_name = opts['model_type']
    loss_type = opts['loss_type']
    num_channels = opts["num_channels"]
    num_classes = opts["num_classes"]
    encoder_type = opts['encoder_type']

    if model_name == 'PVT_MRE':
        if loss_type == 'single':
            model = PVTMRE_SingleLoss(encoder_name=encoder_type, in_channels=num_channels, classes=num_classes)
        elif loss_type == 'up_sampling':
            model = PVTMRE_Up_Sampling_Loss(encoder_name=encoder_type, in_channels=num_channels, classes=num_classes)
        elif loss_type == 'multi_layer':
            model = PVTMRE_Multi_Layer_Loss(encoder_name=encoder_type, in_channels=num_channels, classes=num_classes)
        elif loss_type == 'hierarchical_fusing':
            model = PVTMRE_Hierarchical_Fusing_Loss(encoder_name=encoder_type, in_channels=num_channels, classes=num_classes)
        else:
            raise ValueError('Invalid loss criterion')
    elif model_name == 'UNet_MRE':
        if loss_type == 'single':
            model = UNetMRE_Single_Loss(n_channels=num_channels, n_classes=num_classes)
        elif loss_type == 'up_sampling':
            model = UNetMRE_Up_Sampling_Loss(n_channels=num_channels, n_classes=num_classes)
        elif loss_type == 'multi_layer':
            model = UNetMRE_Multi_Layer_Loss(n_channels=num_channels, n_classes=num_classes)
        elif loss_type == 'hierarchical_fusing':
            model = UNetMRE_Hierarchical_Fusing_Loss(n_channels=num_channels, n_classes=num_classes)
        else:
            raise ValueError('Invalid loss criterion')
    return model