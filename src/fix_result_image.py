"""
결과 이미지를 올바른 크기로 리사이즈하고 분할하는 스크립트
이미지가 세로로 길게 저장된 경우를 처리
"""
import os
import sys
from PIL import Image
import numpy as np

# PIL 이미지 크기 제한 해제
Image.MAX_IMAGE_PIXELS = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def fix_and_resize_image(input_path, output_path=None, target_height_per_row=512):
    """
    세로로 긴 이미지를 올바른 비율로 리사이즈
    
    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        target_height_per_row: 각 행의 목표 높이
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
    
    print(f"[Info] 각 셀 크기: {cell_width} x {cell_height} 픽셀")
    
    # 목표 크기 계산 (비율 유지)
    aspect_ratio = cell_width / cell_height
    target_cell_height = target_height_per_row
    target_cell_width = int(target_cell_height * aspect_ratio)
    
    # 전체 이미지 크기
    new_width = target_cell_width * num_cols
    new_height = target_cell_height * num_rows
    
    print(f"[Info] 리사이즈 목표 크기: {new_width} x {new_height} 픽셀")
    print(f"[Info] 각 셀 크기: {target_cell_width} x {target_cell_height} 픽셀")
    
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
            resized_cell = cell_img.resize((target_cell_width, target_cell_height), Image.LANCZOS)
            row_images.append(resized_cell)
        
        # 행을 가로로 연결
        row_img = Image.new('RGB', (new_width, target_cell_height))
        for col_idx, cell_img in enumerate(row_images):
            row_img.paste(cell_img, (col_idx * target_cell_width, 0))
        resized_cells.append(row_img)
    
    # 모든 행을 세로로 연결
    final_img = Image.new('RGB', (new_width, new_height))
    for row_idx, row_img in enumerate(resized_cells):
        final_img.paste(row_img, (0, row_idx * target_cell_height))
    
    # 출력 경로 설정
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(config.output_path, f"{base_name}_fixed.png")
    
    print(f"[Info] 저장 중: {output_path}")
    final_img.save(output_path, "PNG", optimize=True)
    
    new_size = os.path.getsize(output_path) / (1024*1024)
    print(f"[Info] 리사이즈 완료!")
    print(f"[Info] 새 파일 크기: {new_size:.2f} MB")
    print(f"[Info] 크기 감소: {os.path.getsize(input_path) / (1024*1024) - new_size:.2f} MB")
    print(f"[Info] 출력 파일: {output_path}")
    
    return output_path

def split_into_individual_images(input_path, output_dir=None):
    """
    결과 이미지를 개별 이미지로 분할
    
    Args:
        input_path: 입력 이미지 경로
        output_dir: 출력 디렉토리
    """
    if not os.path.exists(input_path):
        print(f"[Error] 파일을 찾을 수 없습니다: {input_path}")
        return
    
    print(f"[Info] 이미지 로드 중: {input_path}")
    img = Image.open(input_path)
    
    width, height = img.size
    num_rows = 4
    num_cols = 8
    
    cell_width = width // num_cols
    cell_height = height // num_rows
    
    print(f"[Info] 원본 크기: {width} x {height} 픽셀")
    print(f"[Info] 셀 크기: {cell_width} x {cell_height} 픽셀")
    
    # 출력 디렉토리 설정
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(config.output_path, f"{base_name}_individual")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 행별 설명
    row_labels = [
        "original_no_smile",
        "scale_1.0_slight_smile",
        "scale_2.0_moderate_smile",
        "scale_3.0_strong_smile"
    ]
    
    print(f"[Info] 이미지 분할 중...")
    count = 0
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * cell_width
            top = row * cell_height
            right = left + cell_width
            bottom = top + cell_height
            
            cell_img = img.crop((left, top, right, bottom))
            
            # 파일명: row{row+1}_col{col+1}_{label}.png
            filename = f"row{row+1}_col{col+1}_{row_labels[row]}.png"
            filepath = os.path.join(output_dir, filename)
            cell_img.save(filepath, "PNG")
            
            count += 1
            if count % 8 == 0:
                print(f"  처리 중: {count}/{num_rows * num_cols} ({row_labels[row]} 행 완료)")
    
    print(f"[Info] 분할 완료!")
    print(f"[Info] 출력 디렉토리: {output_dir}")
    print(f"[Info] 총 {num_rows * num_cols}개 이미지 생성됨")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='결과 이미지 수정 및 분할')
    parser.add_argument('--input', type=str, 
                       default='output/stylegan_latent_control_smiling.png',
                       help='입력 이미지 경로')
    parser.add_argument('--mode', type=str, choices=['resize', 'split', 'both'],
                       default='both', help='처리 모드')
    parser.add_argument('--height', type=int, default=512,
                       help='각 행의 목표 높이 (기본값: 512)')
    
    args = parser.parse_args()
    
    if args.mode in ['resize', 'both']:
        print("=" * 60)
        print("이미지 리사이즈 (올바른 비율)")
        print("=" * 60)
        fix_and_resize_image(args.input, target_height_per_row=args.height)
        print()
    
    if args.mode in ['split', 'both']:
        print("=" * 60)
        print("이미지 분할")
        print("=" * 60)
        split_into_individual_images(args.input)
        print()

