"""
CelebA 이미지를 정사각형으로 변환하는 스크립트
StyleGAN2-ADA는 정사각형 이미지를 요구함
"""
import os
from PIL import Image
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def prepare_square_images(source_dir, dest_dir, size=128):
    """
    CelebA 이미지를 정사각형으로 변환
    
    Args:
        source_dir: 원본 이미지 디렉토리
        dest_dir: 변환된 이미지 저장 디렉토리
        size: 최종 이미지 크기
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    # 이미지 파일 목록
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"[Prepare] Converting {len(image_files)} images to {size}x{size} square...")
    
    for img_file in tqdm(image_files):
        try:
            img_path = os.path.join(source_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            
            # 정사각형으로 크롭 (중앙 크롭)
            width, height = img.size
            min_dim = min(width, height)
            
            # 중앙 크롭
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            
            img_cropped = img.crop((left, top, right, bottom))
            
            # 리사이즈
            img_resized = img_cropped.resize((size, size), Image.LANCZOS)
            
            # 저장
            dest_path = os.path.join(dest_dir, img_file)
            img_resized.save(dest_path, 'JPEG', quality=95)
            
        except Exception as e:
            print(f"[Warning] Failed to process {img_file}: {e}")
            continue
    
    print(f"[Prepare] Conversion completed! Images saved to {dest_dir}")

if __name__ == '__main__':
    source_dir = config.celebA_image_path
    dest_dir = os.path.join(config.project_dir, 'dataset', 'celebA', 'img_align_celeba_square_128')
    
    prepare_square_images(source_dir, dest_dir, size=128)

