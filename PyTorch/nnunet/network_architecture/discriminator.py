from torch import nn
# import EffNet
#
#
# def get_fc_discriminator(num_classes, ndf=64):
#     #(128,128,3)，（in_size - K + 2P）/ S +1
#     ENCODER = 'efficientnet-b2'
#     ENCODER_WEIGHTS = 'imagenet'
#     ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
#     DECODER_ATTENTION_TYPE = 'scse'
#     DEVICE = 'cuda'
#
#     # create segmentation model with pretrained encoder and auxiliary classification branch
#     aux_params = dict(
#         pooling='avg',  # one of 'avg', 'max'
#         dropout=0.4,  # dropout ratio, default is None
#         activation='sigmoid',  # activation function, default is None
#         classes=2,  # define number of output labels
#     )
#     discri_model = EffNet.GetModelFromName(
#         encoder_name=ENCODER,
#         encoder_weights=ENCODER_WEIGHTS,
#         activation=ACTIVATION,
#         aux_params=aux_params)
#     return discri_model


# def get_fc_discriminator(num_classes, ndf=64):
#     #(128,128,3)，（in_size - K + 2P）/ S +1
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1), #(64,64,64)
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1), #(32,32,128)
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1), #(16,16,256)
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1), #(8,8,512)
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1), #(4,4,1)
#     )

def get_fc_discriminator(num_classes, threeD=False, ndf=64):
    if threeD:
        # (3, 160, 128, 112)，（in_size - K + 2P）/ S +1
        return nn.Sequential(
            nn.Conv3d(num_classes, ndf, kernel_size=3, stride=1, padding=1),  # (64, 160, 128, 112)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf, ndf*2, kernel_size=3, stride=1, padding=1), # (128, 160, 128, 112)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf*2, ndf * 4, kernel_size=4, stride=2, padding=1), # (256, 80, 64, 56) (x-4+2)/2+1=x/2
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1), #(512, 80, 64, 56)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf * 8, ndf * 4, kernel_size=3, stride=1, padding=1), #(256, 80, 64, 56)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf * 4, ndf, kernel_size=3, stride=1, padding=1),  # (64, 80, 64, 56)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf, 1, kernel_size=3, stride=1, padding=1), # (1, 80, 64, 56)
        )
    else:
        #(128,128,3)，（in_size - K + 2P）/ S +1
        return nn.Sequential(
            nn.Conv2d(num_classes, ndf, kernel_size=3, stride=1, padding=1),  # (128,128,64)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=1, padding=1), #(128,128,128)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf * 4, kernel_size=4, stride=2, padding=1), #(64,64,256)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1), #(64,64,512)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 4, kernel_size=3, stride=1, padding=1), #(32,32,256)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf, kernel_size=3, stride=1, padding=1),  # (32,32,64)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, 1, kernel_size=3, stride=1, padding=1), #(32,32,1)
        )
