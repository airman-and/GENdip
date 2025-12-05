"""
결과 이미지를 확인 가능한 크기로 재생성하는 스크립트
원본 이미지가 세로로 길게 저장된 경우를 처리
"""
import os
import sys
from PIL import Image
import numpy as np

# PIL 이미지 크기 제한 해제
Image.MAX_IMAGE_PIXELS = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def create_viewable_image(input_path, output_path=None, image_size=128):
    """
    결과 이미지를 확인 가능한 크기로 재생성
    
    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        image_size: 각 이미지의 크기 (정사각형)
    """
    if not os.path.exists(input_path):
        print(f"[Error] 파일을 찾을 수 없습니다: {input_path}")
        return
    
    print(f"[Info] 이미지 로드 중: {input_path}")
    img = Image.open(input_path)
    
    original_width, original_height = img.size
    print(f"[Info] 원본 크기: {original_width} x {original_height} 픽셀")
    
    # 이미지가 4행 8열로 구성되어 있다고 가정
    num_rows = 4
    num_cols = 8
    
    # 각 셀의 원본 크기
    cell_width = original_width // num_cols
    cell_height = original_height // num_rows
    
    print(f"[Info] 각 셀 원본 크기: {cell_width} x {cell_height} 픽셀")
    
    # 각 이미지를 정사각형으로 리사이즈
    target_size = image_size
    
    # 전체 이미지 크기
    new_width = target_size * num_cols
    new_height = target_size * num_rows
    
    print(f"[Info] 목표 크기: {new_width} x {new_height} 픽셀")
    print(f"[Info] 각 이미지 크기: {target_size} x {target_size} 픽셀")
    
    # 이미지를 셀 단위로 리사이즈하여 재구성
    resized_cells = []
    for row in range(num_rows):
        row_images = []
        for col in range(num_cols):
            left = col * cell_width
            top = row * cell_height
            right = left + cell_width
            bottom = top + cell_height
            
            cell_img = img.crop((left, top, right, bottom))
            # 정사각형으로 리사이즈 (비율 유지하며 중앙 크롭)
            resized_cell = cell_img.resize((target_size, target_size), Image.LANCZOS)
            row_images.append(resized_cell)
        
        # 행을 가로로 연결
        row_img = Image.new('RGB', (new_width, target_size))
        for col_idx, cell_img in enumerate(row_images):
            row_img.paste(cell_img, (col_idx * target_size, 0))
        resized_cells.append(row_img)
    
    # 모든 행을 세로로 연결
    final_img = Image.new('RGB', (new_width, new_height))
    for row_idx, row_img in enumerate(resized_cells):
        final_img.paste(row_img, (0, row_idx * target_size))
    
    # 출력 경로 설정
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(config.output_path, f"{base_name}_viewable.png")
    
    print(f"[Info] 저장 중: {output_path}")
    final_img.save(output_path, "PNG", optimize=True)
    
    new_size = os.path.getsize(output_path) / (1024*1024)
    print(f"[Info] 재생성 완료!")
    print(f"[Info] 새 파일 크기: {new_size:.2f} MB")
    print(f"[Info] 크기 감소: {os.path.getsize(input_path) / (1024*1024) - new_size:.2f} MB")
    print(f"[Info] 출력 파일: {output_path}")
    print()
    print("=" * 60)
    print("이미지 구성:")
    print("=" * 60)
    print("Row 1: 원본 이미지 (미소 없음)")
    print("Row 2: Scale 1.0 (약한 미소)")
    print("Row 3: Scale 2.0 (중간 미소)")
    print("Row 4: Scale 3.0 (강한 미소)")
    print("=" * 60)
    
    return output_path

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='결과 이미지를 확인 가능한 크기로 재생성')
    parser.add_argument('--input', type=str, 
                       default='output/stylegan_latent_control_smiling.png',
                       help='입력 이미지 경로')
    parser.add_argument('--output', type=str, default=None,
                       help='출력 이미지 경로')
    parser.add_argument('--size', type=int, default=256,
                       help='각 이미지의 크기 (기본값: 256)')
    
    args = parser.parse_args()
    
    create_viewable_image(args.input, args.output, args.size)

