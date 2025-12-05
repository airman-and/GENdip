"""
조각난 이미지를 올바르게 재구성하는 스크립트
원본 이미지가 잘못 저장된 경우를 처리
"""
import os
import sys
from PIL import Image
import numpy as np
import torch
from torchvision.utils import make_grid, save_image

# PIL 이미지 크기 제한 해제
Image.MAX_IMAGE_PIXELS = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def reconstruct_image_from_broken(input_path, output_path=None, image_size=128, num_rows=4, num_cols=8):
    """
    조각난 이미지를 올바르게 재구성
    
    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        image_size: 각 이미지의 크기
        num_rows: 행 개수
        num_cols: 열 개수
    """
    if not os.path.exists(input_path):
        print(f"[Error] 파일을 찾을 수 없습니다: {input_path}")
        return
    
    print(f"[Info] 이미지 로드 중: {input_path}")
    img = Image.open(input_path)
    
    original_width, original_height = img.size
    print(f"[Info] 원본 크기: {original_width} x {original_height} 픽셀")
    
    # 원본 이미지를 numpy 배열로 변환
    img_array = np.array(img)
    print(f"[Info] 이미지 배열 shape: {img_array.shape}")
    
    # 이미지가 세로로 길게 저장된 경우를 처리
    # 각 셀의 실제 크기 계산
    cell_width = original_width // num_cols
    cell_height = original_height // num_rows
    
    print(f"[Info] 각 셀 크기: {cell_width} x {cell_height} 픽셀")
    
    # 각 셀을 추출하고 리사이즈
    cells = []
    for row in range(num_rows):
        row_cells = []
        for col in range(num_cols):
            left = col * cell_width
            top = row * cell_height
            right = left + cell_width
            bottom = top + cell_height
            
            # 셀 추출
            cell_img = img.crop((left, top, right, bottom))
            
            # 이미지가 세로로 매우 길 경우, 중앙 부분만 사용
            if cell_height > cell_width * 10:  # 세로가 가로의 10배 이상이면
                # 중앙 부분만 크롭
                center_y = cell_height // 2
                crop_size = min(cell_width, cell_height // 10)
                top_crop = max(0, center_y - crop_size // 2)
                bottom_crop = min(cell_height, center_y + crop_size // 2)
                cell_img = cell_img.crop((0, top_crop, cell_width, bottom_crop))
            
            # 정사각형으로 리사이즈
            cell_img = cell_img.resize((image_size, image_size), Image.LANCZOS)
            row_cells.append(cell_img)
        
        cells.append(row_cells)
    
    # 이미지를 올바르게 재구성
    print(f"[Info] 이미지 재구성 중...")
    
    # 각 행을 가로로 연결
    row_images = []
    for row_cells in cells:
        row_img = Image.new('RGB', (image_size * num_cols, image_size))
        for col_idx, cell_img in enumerate(row_cells):
            row_img.paste(cell_img, (col_idx * image_size, 0))
        row_images.append(row_img)
    
    # 모든 행을 세로로 연결
    final_img = Image.new('RGB', (image_size * num_cols, image_size * num_rows))
    for row_idx, row_img in enumerate(row_images):
        final_img.paste(row_img, (0, row_idx * image_size))
    
    # 출력 경로 설정
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(config.output_path, f"{base_name}_reconstructed.png")
    
    print(f"[Info] 저장 중: {output_path}")
    final_img.save(output_path, "PNG", optimize=True)
    
    new_size = os.path.getsize(output_path) / (1024*1024)
    print(f"[Info] 재구성 완료!")
    print(f"[Info] 새 파일 크기: {new_size:.2f} MB")
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

def reconstruct_from_tensor_save(input_path, output_path=None, image_size=128):
    """
    원본 텐서 저장 방식을 고려하여 재구성
    save_image는 이미지를 가로로 배치하므로, 세로로 길게 저장된 경우
    실제로는 이미지가 세로로 쌓여있을 수 있음
    """
    if not os.path.exists(input_path):
        print(f"[Error] 파일을 찾을 수 없습니다: {input_path}")
        return
    
    print(f"[Info] 이미지 로드 중: {input_path}")
    img = Image.open(input_path)
    
    original_width, original_height = img.size
    print(f"[Info] 원본 크기: {original_width} x {original_height} 픽셀")
    
    # save_image는 nrow=8로 저장했으므로
    # 실제로는 이미지가 가로 8개씩 배치되어야 함
    # 하지만 세로로 매우 길게 저장되었다는 것은
    # 이미지가 세로로 쌓여있을 가능성이 있음
    
    # 각 이미지의 예상 크기 (128x128)
    expected_image_size = image_size
    num_cols = 8
    num_rows = 4
    
    # 실제 셀 크기
    cell_width = original_width // num_cols
    cell_height = original_height // num_rows
    
    print(f"[Info] 셀 크기: {cell_width} x {cell_height}")
    
    # 이미지가 세로로 매우 길 경우, 각 셀의 중앙 부분만 사용
    cells = []
    for row in range(num_rows):
        row_cells = []
        for col in range(num_cols):
            left = col * cell_width
            top = row * cell_height
            right = left + cell_width
            bottom = top + cell_height
            
            cell_img = img.crop((left, top, right, bottom))
            
            # 세로가 너무 길면 중앙 부분만 사용
            if cell_height > cell_width * 2:
                # 중앙에서 정사각형 크롭
                center_y = cell_height // 2
                crop_size = cell_width
                top_crop = max(0, center_y - crop_size // 2)
                bottom_crop = min(cell_height, center_y + crop_size // 2)
                cell_img = cell_img.crop((0, top_crop, cell_width, bottom_crop))
            
            # 정사각형으로 리사이즈
            cell_img = cell_img.resize((expected_image_size, expected_image_size), Image.LANCZOS)
            row_cells.append(cell_img)
        
        cells.append(row_cells)
    
    # 재구성
    final_width = expected_image_size * num_cols
    final_height = expected_image_size * num_rows
    
    final_img = Image.new('RGB', (final_width, final_height))
    
    for row_idx, row_cells in enumerate(cells):
        for col_idx, cell_img in enumerate(row_cells):
            x = col_idx * expected_image_size
            y = row_idx * expected_image_size
            final_img.paste(cell_img, (x, y))
    
    # 출력 경로 설정
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(config.output_path, f"{base_name}_reconstructed.png")
    
    print(f"[Info] 저장 중: {output_path}")
    final_img.save(output_path, "PNG", optimize=True)
    
    new_size = os.path.getsize(output_path) / (1024*1024)
    print(f"[Info] 재구성 완료!")
    print(f"[Info] 새 파일 크기: {new_size:.2f} MB")
    print(f"[Info] 출력 파일: {output_path}")
    
    return output_path

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='조각난 이미지 재구성')
    parser.add_argument('--input', type=str, 
                       default='output/stylegan_latent_control_smiling.png',
                       help='입력 이미지 경로')
    parser.add_argument('--output', type=str, default=None,
                       help='출력 이미지 경로')
    parser.add_argument('--size', type=int, default=256,
                       help='각 이미지의 크기 (기본값: 256)')
    
    args = parser.parse_args()
    
    reconstruct_from_tensor_save(args.input, args.output, args.size)

