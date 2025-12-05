"""
실험 진행 상황 확인 스크립트
"""
import os
import glob
import subprocess
from datetime import datetime

def check_running_processes():
    """실행 중인 실험 프로세스 확인"""
    print("=" * 70)
    print("실행 중인 실험 프로세스")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        
        lines = result.stdout.split('\n')
        main_stylegan_processes = [l for l in lines if 'main_stylegan.py' in l and 'grep' not in l]
        
        if main_stylegan_processes:
            print(f"실행 중인 프로세스: {len(main_stylegan_processes)}개\n")
            for line in main_stylegan_processes:
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    cmd = ' '.join(parts[10:])
                    # 속성 이름 추출
                    attr = 'Unknown'
                    if '--attr' in cmd:
                        idx = cmd.index('--attr')
                        if idx + 1 < len(cmd.split()):
                            attr = cmd.split()[idx + 1]
                    print(f"  PID: {pid:>6} | CPU: {cpu:>5}% | MEM: {mem:>5}% | 속성: {attr}")
        else:
            print("실행 중인 프로세스가 없습니다.")
    except Exception as e:
        print(f"프로세스 확인 중 오류: {e}")
    
    print()

def check_log_files():
    """로그 파일 확인"""
    print("=" * 70)
    print("최근 로그 파일")
    print("=" * 70)
    
    log_files = glob.glob('output/stylegan_attribute_control_*.log')
    log_files.sort(key=os.path.getmtime, reverse=True)
    
    if log_files:
        print(f"총 {len(log_files)}개 로그 파일\n")
        for log_file in log_files[:10]:  # 최근 10개만
            attr_name = os.path.basename(log_file).replace('stylegan_attribute_control_', '').replace('.log', '')
            mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
            size = os.path.getsize(log_file) / (1024 * 1024)  # MB
            
            # 로그 파일에서 완료 여부 확인
            status = "실행 중"
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if 'Results saved' in content or 'Individual images saved' in content:
                        status = "✅ 완료"
                    elif 'Error' in content[-1000:]:
                        status = "❌ 오류"
            except:
                pass
            
            print(f"  {attr_name:30s} | {mtime.strftime('%Y-%m-%d %H:%M:%S'):19s} | {size:6.2f}MB | {status}")
    else:
        print("로그 파일이 없습니다.")
    
    print()

def check_result_files():
    """결과 파일 확인"""
    print("=" * 70)
    print("생성된 결과 파일")
    print("=" * 70)
    
    # 전체 결과 이미지
    result_images = glob.glob('output/stylegan_latent_control_*.png')
    result_images = [f for f in result_images if 'individual' not in f and 'viewable' not in f and 'fixed' not in f and 'resized' not in f and 'reconstructed' not in f]
    
    # 개별 결과 폴더
    result_dirs = [d for d in glob.glob('output/stylegan_latent_control_*_individual') if os.path.isdir(d)]
    
    if result_images or result_dirs:
        print(f"전체 결과 이미지: {len(result_images)}개")
        for img in sorted(result_images):
            attr_name = os.path.basename(img).replace('stylegan_latent_control_', '').replace('.png', '')
            mtime = datetime.fromtimestamp(os.path.getmtime(img))
            size = os.path.getsize(img) / 1024  # KB
            print(f"  ✅ {attr_name:30s} | {mtime.strftime('%Y-%m-%d %H:%M:%S'):19s} | {size:8.2f}KB")
        
        print(f"\n개별 결과 폴더: {len(result_dirs)}개")
        for dir_path in sorted(result_dirs):
            attr_name = os.path.basename(dir_path).replace('stylegan_latent_control_', '').replace('_individual', '')
            files = glob.glob(os.path.join(dir_path, '*.png'))
            if files:
                mtime = datetime.fromtimestamp(os.path.getmtime(files[0]))
                print(f"  ✅ {attr_name:30s} | {mtime.strftime('%Y-%m-%d %H:%M:%S'):19s} | {len(files)}개 파일")
    else:
        print("결과 파일이 없습니다.")
    
    print()

def check_gpu_usage():
    """GPU 사용률 확인"""
    print("=" * 70)
    print("GPU 사용률")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id = parts[0]
                    util = parts[1]
                    mem_used = int(parts[2])
                    mem_total = int(parts[3])
                    mem_percent = (mem_used / mem_total) * 100
                    print(f"  GPU {gpu_id}: 사용률 {util:>3}% | 메모리 {mem_used:>6}MB / {mem_total:>6}MB ({mem_percent:5.1f}%)")
        else:
            print("GPU 정보를 가져올 수 없습니다.")
    except Exception as e:
        print(f"GPU 확인 중 오류: {e}")
    
    print()

def main():
    """메인 함수"""
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("\n" + "=" * 70)
    print("StyleGAN 속성 제어 실험 상태 확인")
    print("=" * 70)
    print(f"확인 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    check_running_processes()
    check_gpu_usage()
    check_log_files()
    check_result_files()
    
    print("=" * 70)
    print("확인 완료!")
    print("=" * 70)

if __name__ == '__main__':
    main()

