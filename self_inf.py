import os
import torch
import logging
import argparse
from utils import get_root_logger, get_time_str
from models import build_model
from PIL import Image
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--pretrained_model', type=str, default=r'./pretrained_model/DualDn_Big.pth', help='Path to pretrained model.')
    parser.add_argument('--output_path', type=str, default='./output.png', help='Path to save the processed image.')
    args = parser.parse_args()
    return args

def load_image(image_path):
    """Load an image and preprocess it."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def save_image(tensor, output_path):
    """Save a tensor as an image."""
    image = transforms.ToPILImage()(tensor.squeeze(0))
    image.save(output_path)

def main():
    args = parse_options()

    # Set up logging
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)

    # Prepare model options
    model_options = {
        'model_type': 'DualDn_Model',  # 실제 등록된 이름으로 수정
        'path': {'pretrain_network': args.pretrained_model},  # 사전 학습된 가중치 경로
        # 필요한 경우 추가 옵션 포함
    }

    # Load model
    logger.info('Loading model...')
    model = build_model(model_options)
    model.eval()

    # Load image
    logger.info(f'Loading image from {args.image_path}...')
    image = load_image(args.image_path)
    if torch.cuda.is_available():
        image = image.cuda()

    # Process image
    logger.info('Processing image...')
    with torch.no_grad():
        output = model(image)

    # Save the output
    logger.info(f'Saving output to {args.output_path}...')
    save_image(output, args.output_path)
    logger.info('Processing complete.')

if __name__ == '__main__':
    main()