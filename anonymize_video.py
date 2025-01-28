import argparse
import os
import time

import cv2
import torch

from arcface_model.iresnet import iresnet100
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from models.config_sr import TestOptions
from models.pix2pix_model import Pix2PixModel
from network.AEI_Net import AEI_Net
from utils.inference.core import model_anonymize
from utils.inference.image_processing import crop_face, get_final_image
from utils.inference.video_processing import (
    add_audio_from_another_video,
    face_enhancement,
    get_final_video,
    get_target,
    read_video,
)
from utils.startup import anon_parser


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


def main(args):
    app, G, netArc, handler, model = init_models(args)

    assert args.image_to_image is False

    full_frames, fps = read_video(args.target_video)
    # source = []
    # frame_idx = 0
    # while len(source) == 0:
    #     if frame_idx >= len(full_frames):
    #         raise Exception("No faces found in video!")
    #     try:
    #         img = cv2.imread(full_frames[frame_idx])
    #         img = crop_face(img, app, args.crop_size)[0]
    #         source.append(img[:, :, ::-1])
    #     except:
    #         frame_idx += 1

    # get target faces that are used for swap
    print("List of target paths: ", args.target_faces_paths)
    if not args.target_faces_paths:
        target = get_target(full_frames, app, args.crop_size)
    else:
        target = []
        try:
            for target_faces_path in args.target_faces_paths:
                img = cv2.imread(target_faces_path)
                img = crop_face(img, app, args.crop_size)[0]
                target.append(img)
        except TypeError:
            print("Bad target images!")
            exit()

    start = time.time()
    final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_anonymize(
        full_frames,
        target,
        target,
        netArc,
        G,
        app,
        True,
        similarity_th=args.similarity_th,
        crop_size=args.crop_size,
        BS=args.batch_size,
        epsilon=args.epsilon,
        theta=args.theta,
    )
    if args.use_sr:
        final_frames_list = face_enhancement(final_frames_list, model)

    if not args.image_to_image:
        get_final_video(
            final_frames_list,
            crop_frames_list,
            full_frames,
            tfm_array_list,
            args.out_video_name,
            fps,
            handler,
        )

        add_audio_from_another_video(args.target_video, args.out_video_name, "audio")
        print(f"Video saved with path {args.out_video_name}")
    else:
        result = get_final_image(
            final_frames_list, crop_frames_list, full_frames[0], tfm_array_list, handler
        )
        cv2.imwrite(args.out_image_name, result)
        print(f"Swapped Image saved with path {args.out_image_name}")

    print("Total time: ", time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = anon_parser()

    parser.add_argument(
        "--source_paths",
        default=["examples/images/mark.jpg", "examples/images/elon_musk.jpg"],
        nargs="+",
    )
    parser.add_argument(
        "--target_faces_paths",
        default=[],
        nargs="+",
        help="It's necessary to set the face/faces in the video to which the source face/faces is swapped. You can skip this parametr, and then any face is selected in the target video for swap.",
    )

    # parameters for image to video
    parser.add_argument(
        "--target_video",
        default="examples/videos/nggyup.mp4",
        type=str,
        help="It's necessary for image to video swap",
    )
    parser.add_argument(
        "--out_video_name",
        default="results/result.mp4",
        type=str,
        help="It's necessary for image to video swap",
    )

    # parameters for image to image
    parser.add_argument(
        "--image_to_image",
        default=False,
        type=bool,
        help="True for image to image swap, False for swap on video",
    )

    args = parser.parse_args()
    main(args)
