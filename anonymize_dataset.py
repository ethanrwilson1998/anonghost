import glob
import ntpath
import os

from anonymize_image import anonymize_image
from utils.startup import anon_parser


def main(args):
    input_paths = glob.glob(f"{args.input_dir}//**//*.jpg", recursive=True)

    src_paths, out_paths = [], []

    for i_p in input_paths:
        rel_path = i_p.split(args.input_dir)[1]
        to_gen = f"{args.output_dir}{rel_path}"

        os.makedirs(ntpath.dirname(to_gen), exist_ok=True)

        src_paths.append(i_p)
        out_paths.append(to_gen)

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
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Top-level directory to search for images to anonymize.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Top-level directory to place anonymized images.",
    )
    args = parser.parse_args()
    main(args)
