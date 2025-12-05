"""
이미지를 올바르게 재구성 - 개선 버전
각 셀의 중앙 부분을 정확히 추출하여 재구성
"""
import os
import sys
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def fix_image_properly(input_path, output_path=None, target_size=256):
    """
    조각난 이미지를 올바르게 재구성
    
    원본 이미지가 1042 x 195132로 저장되어 있고,
    각 셀이 130 x 48783인 경우를 처리
    각 셀의 중앙 부분(정사각형)을 추출하여 재구성
    """
    if not os.path.exists(input_path):
        print(f"[Error] 파일을 찾을 수 없습니다: {input_path}")
        return
    
    print(f"[Info] 이미지 로드 중: {input_path}")
    img = Image.open(input_path)
    
    original_width, original_height = img.size
    print(f"[Info] 원본 크기: {original_width} x {original_height} 픽셀")
    
    num_cols = 8
    num_rows = 4
    
    cell_width = original_width // num_cols
    cell_height = original_height // num_rows
    
    print(f"[Info] 각 셀 크기: {cell_width} x {cell_height} 픽셀")
    print(f"[Info] 목표 이미지 크기: {target_size} x {target_size} 픽셀")
    
    # 각 셀에서 중앙 정사각형 부분만 추출
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
            
            # 세로가 매우 긴 경우, 중앙에서 정사각형 크롭
            if cell_height > cell_width:
                # 중앙 Y 좌표
                center_y = cell_height // 2
                # 정사각형 크기 (가로 크기 사용)
                crop_size = cell_width
                # 크롭 영역 계산
                crop_top = max(0, center_y - crop_size // 2)
                crop_bottom = min(cell_height, center_y + crop_size // 2)
                
                # 중앙 부분만 추출
                cell_img = cell_img.crop((0, crop_top, cell_width, crop_bottom))
            
            # 목표 크기로 리사이즈
            cell_img = cell_img.resize((target_size, target_size), Image.LANCZOS)
            row_cells.append(cell_img)
        
        cells.append(row_cells)
    
    # 이미지 재구성
    print(f"[Info] 이미지 재구성 중...")
    
    final_width = target_size * num_cols
    final_height = target_size * num_rows
    
    final_img = Image.new('RGB', (final_width, final_height))
    
    for row_idx, row_cells in enumerate(cells):
        for col_idx, cell_img in enumerate(row_cells):
            x = col_idx * target_size
            y = row_idx * target_size
            final_img.paste(cell_img, (x, y))
    
    # 출력 경로 설정
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(config.output_path, f"{base_name}_fixed_properly.png")
    
    print(f"[Info] 저장 중: {output_path}")
    final_img.save(output_path, "PNG", optimize=True)
    
    new_size = os.path.getsize(output_path) / (1024*1024)
    print(f"[Info] 재구성 완료!")
    print(f"[Info] 새 파일 크기: {new_size:.2f} MB")
    print(f"[Info] 출력 파일: {output_path}")
    print()
    print("=" * 60)
    print("이미지 구성 (4행 × 8열):")
    print("=" * 60)
    print("Row 1: 원본 이미지 8개 (미소 없음)")
    print("Row 2: Scale 1.0 적용 8개 (약한 미소)")
    print("Row 3: Scale 2.0 적용 8개 (중간 미소)")
    print("Row 4: Scale 3.0 적용 8개 (강한 미소)")
    print("=" * 60)
    
    return output_path

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='이미지 올바르게 재구성')
    parser.add_argument('--input', type=str, 
                       default='output/stylegan_latent_control_smiling.png',
                       help='입력 이미지 경로')
    parser.add_argument('--output', type=str, default=None,
                       help='출력 이미지 경로')
    parser.add_argument('--size', type=int, default=256,
                       help='각 이미지의 크기 (기본값: 256)')
    
    args = parser.parse_args()
    
    fix_image_properly(args.input, args.output, args.size)

