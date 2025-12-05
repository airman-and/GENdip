"""
결과 이미지를 작은 크기로 리사이즈하는 스크립트
"""
import os
import sys
from PIL import Image

# PIL 이미지 크기 제한 해제
Image.MAX_IMAGE_PIXELS = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def resize_result_image(input_path, output_path=None, max_size=2048):
    """
    결과 이미지를 작은 크기로 리사이즈
    
    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로 (None이면 자동 생성)
        max_size: 최대 크기 (가로 또는 세로)
    """
    if not os.path.exists(input_path):
        print(f"[Error] 파일을 찾을 수 없습니다: {input_path}")
        return
    
    print(f"[Info] 이미지 로드 중: {input_path}")
    img = Image.open(input_path)
    
    original_size = img.size
    print(f"[Info] 원본 크기: {original_size[0]} x {original_size[1]} 픽셀")
    print(f"[Info] 원본 파일 크기: {os.path.getsize(input_path) / (1024*1024):.2f} MB")
    
    # 비율 유지하며 리사이즈
    if original_size[0] > original_size[1]:
        # 가로가 더 긴 경우
        new_width = max_size
        new_height = int(original_size[1] * (max_size / original_size[0]))
    else:
        # 세로가 더 긴 경우
        new_height = max_size
        new_width = int(original_size[0] * (max_size / original_size[1]))
    
    print(f"[Info] 리사이즈 중: {new_width} x {new_height} 픽셀")
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # 출력 경로 설정
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(config.output_path, f"{base_name}_resized.png")
    
    print(f"[Info] 저장 중: {output_path}")
    resized_img.save(output_path, "PNG", optimize=True)
    
    new_size = os.path.getsize(output_path) / (1024*1024)
    print(f"[Info] 리사이즈 완료!")
    print(f"[Info] 새 파일 크기: {new_size:.2f} MB")
    print(f"[Info] 크기 감소: {os.path.getsize(input_path) / (1024*1024) - new_size:.2f} MB")
    print(f"[Info] 출력 파일: {output_path}")

def split_result_image(input_path, output_dir=None, rows=4, cols=8):
    """
    결과 이미지를 개별 이미지로 분할
    
    Args:
        input_path: 입력 이미지 경로
        output_dir: 출력 디렉토리 (None이면 자동 생성)
        rows: 행 개수
        cols: 열 개수
    """
    if not os.path.exists(input_path):
        print(f"[Error] 파일을 찾을 수 없습니다: {input_path}")
        return
    
    print(f"[Info] 이미지 로드 중: {input_path}")
    img = Image.open(input_path)
    
    width, height = img.size
    cell_width = width // cols
    cell_height = height // rows
    
    print(f"[Info] 원본 크기: {width} x {height} 픽셀")
    print(f"[Info] 셀 크기: {cell_width} x {cell_height} 픽셀")
    
    # 출력 디렉토리 설정
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(config.output_path, f"{base_name}_split")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Info] 이미지 분할 중...")
    for row in range(rows):
        for col in range(cols):
            left = col * cell_width
            top = row * cell_height
            right = left + cell_width
            bottom = top + cell_height
            
            cell_img = img.crop((left, top, right, bottom))
            
            # 파일명: row_col.png
            filename = f"row{row+1}_col{col+1}.png"
            filepath = os.path.join(output_dir, filename)
            cell_img.save(filepath, "PNG")
            
            if (row * cols + col + 1) % 8 == 0:
                print(f"  처리 중: {row * cols + col + 1}/{rows * cols}")
    
    print(f"[Info] 분할 완료!")
    print(f"[Info] 출력 디렉토리: {output_dir}")
    print(f"[Info] 총 {rows * cols}개 이미지 생성됨")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='결과 이미지 리사이즈 또는 분할')
    parser.add_argument('--input', type=str, 
                       default='output/stylegan_latent_control_smiling.png',
                       help='입력 이미지 경로')
    parser.add_argument('--mode', type=str, choices=['resize', 'split', 'both'],
                       default='both', help='처리 모드')
    parser.add_argument('--max-size', type=int, default=2048,
                       help='리사이즈 최대 크기 (기본값: 2048)')
    parser.add_argument('--rows', type=int, default=4,
                       help='행 개수 (분할용, 기본값: 4)')
    parser.add_argument('--cols', type=int, default=8,
                       help='열 개수 (분할용, 기본값: 8)')
    
    args = parser.parse_args()
    
    if args.mode in ['resize', 'both']:
        print("=" * 60)
        print("이미지 리사이즈")
        print("=" * 60)
        resize_result_image(args.input, max_size=args.max_size)
        print()
    
    if args.mode in ['split', 'both']:
        print("=" * 60)
        print("이미지 분할")
        print("=" * 60)
        split_result_image(args.input, rows=args.rows, cols=args.cols)
        print()

