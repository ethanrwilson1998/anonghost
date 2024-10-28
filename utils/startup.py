import argparse
import os

import torch

from arcface_model.iresnet import iresnet100
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from models.config_sr import TestOptions
from models.pix2pix_model import Pix2PixModel
from network.AEI_Net import AEI_Net

# Global vars to check if we've already loaded models
HAS_INITIALIZED = False
MODELS = None


def init_models(args):
    # model for face cropping
    app = Face_detect_crop(name="antelope", root="./insightface_func/models")
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    # main model for generation
    G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512)
    G.eval()
    G.load_state_dict(torch.load(args.G_path, map_location=torch.device("cpu")))
    G = G.cuda()
    G = G.half()

    # arcface model to get face embedding
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load("arcface_model/backbone.pth"))
    netArc = netArc.cuda()
    netArc.eval()

    # model to get face landmarks
    handler = Handler("./coordinate_reg/model/2d106det", 0, ctx_id=0, det_size=640)

    # model to make superres of face, set use_sr=True if you want to use super resolution or use_sr=False if you don't
    if args.use_sr:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.backends.cudnn.benchmark = True
        opt = TestOptions()
        # opt.which_epoch ='10_7'
        model = Pix2PixModel(opt)
        model.netG.train()
    else:
        model = None

    return app, G, netArc, handler, model


def grab_models(args):
    global HAS_INITIALIZED, MODELS
    if not HAS_INITIALIZED:
        MODELS = init_models(args)
        HAS_INITIALIZED = True
    return MODELS


def base_parser():
    parser = argparse.ArgumentParser()

    # Generator params
    parser.add_argument(
        "--G_path",
        default="weights/G_unet_2blocks.pth",
        type=str,
        help="Path to weights for G",
    )
    parser.add_argument(
        "--backbone",
        default="unet",
        const="unet",
        nargs="?",
        choices=["unet", "linknet", "resnet"],
        help="Backbone for attribute encoder",
    )
    parser.add_argument(
        "--num_blocks", default=2, type=int, help="Numbers of AddBlocks at AddResblock"
    )

    parser.add_argument("--batch_size", default=40, type=int)
    parser.add_argument("--crop_size", default=224, type=int, help="Don't change this")
    parser.add_argument(
        "--use_sr",
        default=True,
        type=bool,
        help="True for super resolution on swap images",
    )
    parser.add_argument(
        "--similarity_th",
        default=0.15,
        type=float,
        help="Threshold for selecting a face similar to the target",
    )

    return parser


def anon_parser():
    parser = base_parser()
    parser.add_argument(
        "--epsilon",
        default=1,
        type=float,
        help="Epsilon parameter in privacy operations",
    )
    parser.add_argument(
        "--theta",
        default=0,
        type=float,
        help="Rotation parameter in privacy operations",
    )
    return parser
