import ntpath

import cv2

from utils.inference.core import model_anonymize
from utils.inference.image_processing import crop_face, get_final_image
from utils.inference.video_processing import (
    face_enhancement,
)
from utils.startup import anon_parser, grab_models


def anonymize_image(img_paths, out_paths, args):
    app, G, netArc, handler, model = grab_models(args)

    if not isinstance(img_paths, list):
        img_paths = [img_paths]
    if not isinstance(out_paths, list):
        out_paths = [out_paths]

    for i, (img_path, out_path) in enumerate(zip(img_paths, out_paths)):
        src_img = cv2.imread(img_path)
        src_img = crop_face(src_img, app, args.crop_size)[0]
        source = [src_img[:, :, ::-1]]

        dst_img = cv2.imread(img_path)
        full_frames = [dst_img]
        dst_img = crop_face(dst_img, app, args.crop_size)[0]
        target = [dst_img]

        final_frames_list, crop_frames_list, full_frames, tfm_array_list = (
            model_anonymize(
                full_frames,
                source,
                target,
                netArc,
                G,
                app,
                True,
                similarity_th=args.similarity_th,
                crop_size=args.crop_size,
                BS=args.batch_size,
                epsilon=20,
                theta=0,
            )
        )
        if args.use_sr:
            final_frames_list = face_enhancement(final_frames_list, model)

        result = get_final_image(
            final_frames_list,
            crop_frames_list,
            full_frames[0],
            tfm_array_list,
            handler,
        )
        cv2.imwrite(out_path, result)


def main(args):
    import glob

    src_paths, out_paths = [], []
    test_imgs = glob.glob("examples/images/*.jpg")
    for t in test_imgs:
        src_paths.append(t)
        out_paths.append(
            f"results/{ntpath.basename(t).split('.')[0]}_eps{args.epsilon}_theta{args.theta}.jpg"
        )

    anonymize_image(src_paths, out_paths, args)


if __name__ == "__main__":
    parser = anon_parser()
    args = parser.parse_args()
    main(args)
