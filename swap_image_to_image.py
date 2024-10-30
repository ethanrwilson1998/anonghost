import cv2

from utils.inference.core import model_inference
from utils.inference.image_processing import crop_face, get_final_image
from utils.inference.video_processing import (
    face_enhancement,
)
from utils.startup import base_parser, grab_models


def swap_image_to_image(src_paths, dst_paths, out_paths, args):
    app, G, netArc, handler, model = grab_models(args)

    if not isinstance(src_paths, list):
        src_paths = [src_paths]
    if not isinstance(dst_paths, list):
        dst_paths = [dst_paths]
    if not isinstance(out_paths, list):
        out_paths = [out_paths]

    for i, (src_path, dst_path, out_path) in enumerate(
        zip(src_paths, dst_paths, out_paths)
    ):
        src_img = cv2.imread(src_path)
        src_img = crop_face(src_img, app, args.crop_size)[0]
        source = [src_img[:, :, ::-1]]

        dst_img = cv2.imread(dst_path)
        full_frames = [dst_img]
        dst_img = crop_face(dst_img, app, args.crop_size)[0]
        target = [dst_img]

        final_frames_list, crop_frames_list, full_frames, tfm_array_list = (
            model_inference(
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
        if args.crop_final_result:
            result = crop_face(result, app, args.crop_size)[0]

        cv2.imwrite(out_path, result)


def main(args):
    import glob

    src_paths, dst_paths, out_paths = [], [], []
    test_imgs = glob.glob("examples/images/*.jpg")
    for x, t in enumerate(test_imgs):
        for y, i in enumerate(test_imgs):
            src_paths.append(t)
            dst_paths.append(i)
            out_paths.append(f"results/test_{x}_{y}.jpg")

    swap_image_to_image(src_paths, dst_paths, out_paths, args)


if __name__ == "__main__":
    parser = base_parser()
    args = parser.parse_args()
    main(args)
