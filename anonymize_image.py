import glob
import ntpath

import cv2
import numpy as np

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

    exc_count = 0

    for i, (img_path, out_path) in enumerate(zip(img_paths, out_paths)):
        try:
            src_img = cv2.imread(img_path)
            src_img = crop_face(src_img, app, args.crop_size)[0]
            source = [src_img[:, :, ::-1]]

            dst_img = cv2.imread(img_path)
            full_frames = [dst_img]
            dst_img = crop_face(dst_img, app, args.crop_size)[0]
            target = [dst_img]

            out_grid = []
            for _ in range(args.grid[0]):
                in_grid = []
                for _ in range(args.grid[1]):
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
                            epsilon=args.epsilon,
                            theta=args.theta,
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
                    in_grid.append(result)
                out_grid.append(in_grid)

            if args.grid != [1, 1]:
                for i in range(len(out_grid)):
                    out_grid[i] = np.concatenate(out_grid[i], axis=0)
                cv2.imwrite(out_path, np.concatenate(out_grid, axis=1))
            else:
                cv2.imwrite(out_path, result)
        except Exception as e:
            print(f"Failed on {img_path} - {e}")
            exc_count += 1

    if exc_count > 0:
        print(f"Warning: {exc_count} images failed to process.")


def main(args):
    for eps in [0, 1, 1e1, 1e2, 1e3, 1e4]:
        args.epsilon = eps

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
    parser.add_argument(
        "--grid",
        default=[1, 1],
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        help="If specified, will render as a grid of generated samples.",
    )
    args = parser.parse_args()
    main(args)
