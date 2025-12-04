import os
import argparse
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms.functional as tf

from . import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default='./pretrained/harmonizer.pth',
                        help='Path to the pretrained model')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Absolute paths to directories containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory')
    parser.add_argument('--apply_filter_mask', type=str, default='1,1,0,0,0',
                        help='Comma-separated boolean values (1/0) for which filters to apply: [brightness,contrast,saturation,hue,color]')
    args = parser.parse_args()

    # Parse the apply_filter_mask string into a list of booleans
    filter_mask = [bool(int(x)) for x in args.apply_filter_mask.split(',')]
    if len(filter_mask) != 5:
        raise ValueError("apply_filter_mask must contain exactly 5 values (got {})".format(len(filter_mask)))

    print('\nEvaluation Harmonizer:')
    print('  - Pretrained Model: {}'.format(args.pretrained))
    print('  - Dataset (directory with images): {}'.format(args.dataset_dir))
    print('  - Harmonization applied globally (using full-image mask)')
    print('  - Apply filter mask: {}'.format(filter_mask))

    cuda = torch.cuda.is_available()

    # Create/load the harmonizer model
    harmonizer = model.Enhancer()
    if cuda:
        harmonizer = harmonizer.cuda()
    harmonizer.load_state_dict(torch.load(args.pretrained), strict=True)
    harmonizer.eval()

    # Process each dataset directory
    # List all image files in the given directory
    dataset_dir = args.dataset_dir
    image_files = sorted([f for f in os.listdir(dataset_dir)])
    if not image_files:
        print("No images found in directory: {}".format(dataset_dir))
        exit()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print("\nProcessing dataset: {}".format(dataset_dir))

    pbar = tqdm(image_files, total=len(image_files), unit='image')
    for image_name in pbar:
        input_path = os.path.join(dataset_dir, image_name)
        comp = Image.open(input_path).convert('RGB')

        # Prepare input: composite image and a global mask (all ones)
        _comp = tf.to_tensor(comp)[None, ...]  # shape: (1, 3, H, W)
        _mask = torch.ones(1, 1, _comp.shape[2], _comp.shape[3])  # global mask

        if cuda:
            _comp = _comp.cuda()
            _mask = _mask.cuda()

        # Predict harmonization arguments and restore the image
        with torch.no_grad():
            arguments = harmonizer.predict_arguments(_comp, _mask)
            _harmonized = harmonizer.restore_image(_comp, _mask, arguments, apply_filter_mask=filter_mask)[-1]
        # print(_harmonized.shape)

        # Save harmonized image using the exact same filename
        result_img = tf.to_pil_image(_harmonized.cpu().squeeze(0).clamp(0, 1))
        output_path = os.path.join(output_dir, image_name)
        result_img.save(output_path)
        pbar.set_description('Processed: {}'.format(image_name))

    print("Harmonized images saved in: {}".format(output_dir))