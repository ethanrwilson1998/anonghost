import glob
import ntpath
import os
import traceback

import cv2
import numpy as np

from utils.inference.core import model_anonymize
from utils.inference.image_processing import crop_face, get_final_image
from utils.inference.video_processing import (
    face_enhancement,
)
from utils.startup import anon_parser, grab_models


def anonymize_image(img_paths, out_paths, args):
    np.random.seed(420)
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
            for g0 in range(args.grid[0]):
                in_grid = []
                for g1 in range(args.grid[1]):
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
                    in_grid.append(
                        cv2.resize(src_img, (result.shape[1], result.shape[0]))
                        if g0 == 1 and g1 == 1
                        else result
                    )
                out_grid.append(in_grid)

            if args.grid != [1, 1]:
                for i in range(len(out_grid)):
                    out_grid[i] = np.concatenate(out_grid[i], axis=0)
                cv2.imwrite(out_path, np.concatenate(out_grid, axis=1))
            else:
                cv2.imwrite(out_path, result)
        except Exception as e:
            print(f"Failed on {img_path} - {e}")
            print(traceback.format_exc())
            exc_count += 1

    if exc_count > 0:
        print(f"Warning: {exc_count} images failed to process.")


def main(args):
    for theta in range(0, 180):
        args.epsilon = -1
        args.theta = theta

        src_paths, out_paths = [], []
        if args.image_dir[-4] in [".jpg", ".png"]:
            test_imgs = [args.image_dir]
        else:
            test_imgs = glob.glob(f"{args.image_dir}/*.jpg") + glob.glob(
                f"{args.image_dir}/*.png"
            )
        for t in test_imgs:
            src_paths.append(t)
            os.makedirs(f"results/{ntpath.basename(t).split('.')[0]}/", exist_ok=True)
            out_paths.append(f"results/{ntpath.basename(t).split('.')[0]}/{theta}.jpg")
        anonymize_image(src_paths, out_paths, args)

    # src_paths, out_paths = [], []
    # if args.image_dir[-4] in [".jpg", ".png"]:
    #     test_imgs = [args.image_dir]
    # else:
    #     test_imgs = glob.glob(f"{args.image_dir}/*.jpg") + glob.glob(
    #         f"{args.image_dir}/*.png"
    #     )
    # for t in test_imgs:
    #     src_paths.append(t)
    #     out_paths.append(
    #         f"results/{ntpath.basename(t).split('.')[0]}_eps{args.epsilon}_theta{args.theta}.jpg"
    #     )

    # anonymize_image(src_paths, out_paths, args)


if __name__ == "__main__":
    parser = anon_parser()
    parser.add_argument(
        "--image_dir",
        default="examples/images",
        type=str,
        help="Which directory to anonymize.",
    )
    parser.add_argument(
        "--grid",
        default=[1, 1],
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        help="If specified, will render as a grid of generated samples.",
    )
    args = parser.parse_args()
    main(args)

# generate grid videos with this:
# ffmpeg -framerate 15 -i %d.jpg -c:v libx264 -crf 1 -vf scale=1536:1536 -pix_fmt yuv420p -vb 100M out.mp4
