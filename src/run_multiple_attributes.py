"""
여러 속성에 대해 한번에 실험을 실행하는 스크립트
"""
import subprocess
import sys
import os

# CelebA 주요 속성 목록
ATTRIBUTES = [
    'Smiling',           # 웃음
    'Eyeglasses',        # 안경
    'Male',             # 남성
    'Young',            # 젊음
    'Blond_Hair',       # 금발
    'Black_Hair',       # 검은 머리
    'Wearing_Hat',      # 모자 착용
    'Wearing_Lipstick', # 립스틱
    'Mustache',         # 수염
    'Goatee',           # 염소 수염
    'Bald',             # 대머리
    'Attractive',       # 매력적
    'Heavy_Makeup',     # 진한 화장
    'Bangs',            # 앞머리
    'Arched_Eyebrows',  # 아치형 눈썹
]

def run_experiment(attr_name, background=False):
    """단일 속성에 대한 실험 실행"""
    print(f"\n{'='*70}")
    print(f"실험 시작: {attr_name}")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable,
        'src/main_stylegan.py',
        '--attr', attr_name
    ]
    
    # 로그 파일명 생성
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    attr_safe = attr_name.lower().replace(' ', '_')
    log_file = f'output/stylegan_attribute_control_{attr_safe}_{timestamp}.log'
    
    print(f"명령어: {' '.join(cmd)}")
    print(f"로그 파일: {log_file}")
    
    if background:
        # 백그라운드 실행 (nohup)
        nohup_cmd = ['nohup'] + cmd + ['>', log_file, '2>&1', '&']
        print(f"백그라운드 실행: {' '.join(nohup_cmd)}")
        try:
            # nohup을 subprocess로 실행
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            print(f"✅ {attr_name} 실험 시작됨 (PID: {process.pid})")
            return process.pid
        except Exception as e:
            print(f"❌ {attr_name} 실험 시작 실패: {e}")
            return None
    else:
        # 포그라운드 실행
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            
            if result.returncode == 0:
                print(f"✅ {attr_name} 실험 완료")
            else:
                print(f"❌ {attr_name} 실험 실패 (코드: {result.returncode})")
            
            return result.returncode == 0
        except Exception as e:
            print(f"❌ {attr_name} 실험 중 오류: {e}")
            return False

def main():
    """여러 속성에 대해 실험 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run StyleGAN attribute control experiments for multiple attributes')
    parser.add_argument(
        '--attrs',
        nargs='+',
        default=None,
        help='List of attributes to experiment with (default: all recommended attributes)'
    )
    parser.add_argument(
        '--start-from',
        type=str,
        default=None,
        help='Start from this attribute (useful for resuming)'
    )
    parser.add_argument(
        '--background',
        action='store_true',
        help='Run experiments in background (sequential, not parallel)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run experiments in parallel (one per GPU, if available)'
    )
    
    args = parser.parse_args()
    
    # 실험할 속성 목록 결정
    if args.attrs:
        attributes_to_run = args.attrs
    else:
        attributes_to_run = ATTRIBUTES
    
    # 시작 위치 결정
    if args.start_from:
        try:
            start_idx = attributes_to_run.index(args.start_from)
            attributes_to_run = attributes_to_run[start_idx:]
            print(f"'{args.start_from}'부터 시작합니다...")
        except ValueError:
            print(f"경고: '{args.start_from}' 속성을 찾을 수 없습니다. 처음부터 시작합니다.")
    
    print(f"\n{'='*70}")
    print(f"총 {len(attributes_to_run)}개 속성에 대한 실험을 시작합니다")
    if args.background:
        print("백그라운드 모드: 순차적으로 실행됩니다")
    elif args.parallel:
        print("병렬 모드: 가능한 만큼 동시 실행됩니다")
    else:
        print("순차 모드: 하나씩 실행됩니다")
    print(f"{'='*70}")
    
    results = {}
    pids = {}
    
    if args.background:
        # 백그라운드 모드: 순차적으로 실행하되 각각 백그라운드로
        for i, attr in enumerate(attributes_to_run, 1):
            print(f"\n[{i}/{len(attributes_to_run)}] {attr}")
            pid = run_experiment(attr, background=True)
            if pid:
                pids[attr] = pid
                results[attr] = 'running'
                # 다음 실험 전에 약간 대기 (시스템 부하 방지)
                import time
                time.sleep(5)
            else:
                results[attr] = False
        
        # PID 저장
        pid_file = 'output/running_experiments.pid'
        with open(pid_file, 'w') as f:
            for attr, pid in pids.items():
                f.write(f"{attr}:{pid}\n")
        print(f"\n실행 중인 프로세스 정보가 {pid_file}에 저장되었습니다")
        
    elif args.parallel:
        # 병렬 모드: GPU 개수만큼 동시 실행
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        print(f"사용 가능한 GPU: {num_gpus}개")
        
        # GPU당 하나씩 실행
        running = {}
        for i, attr in enumerate(attributes_to_run):
            # 실행 중인 프로세스가 GPU 개수만큼이면 대기
            while len(running) >= num_gpus:
                import time
                time.sleep(10)
                # 완료된 프로세스 확인
                completed = []
                for a, p in running.items():
                    try:
                        os.kill(p, 0)  # 프로세스 존재 확인
                    except OSError:
                        completed.append(a)
                for a in completed:
                    del running[a]
            
            pid = run_experiment(attr, background=True)
            if pid:
                running[attr] = pid
                results[attr] = 'running'
        
        # 남은 프로세스 대기
        while running:
            import time
            time.sleep(10)
            completed = []
            for a, p in running.items():
                try:
                    os.kill(p, 0)
                except OSError:
                    completed.append(a)
            for a in completed:
                del running[a]
    else:
        # 순차 모드: 하나씩 실행
        for i, attr in enumerate(attributes_to_run, 1):
            print(f"\n[{i}/{len(attributes_to_run)}] {attr}")
            success = run_experiment(attr, background=False)
            results[attr] = success
    
    # 결과 요약
    print(f"\n{'='*70}")
    print("실험 결과 요약")
    print(f"{'='*70}")
    
    successful = [attr for attr, success in results.items() if success]
    failed = [attr for attr, success in results.items() if not success]
    
    print(f"\n✅ 성공: {len(successful)}개")
    for attr in successful:
        print(f"  - {attr}")
    
    if failed:
        print(f"\n❌ 실패: {len(failed)}개")
        for attr in failed:
            print(f"  - {attr}")
    
    print(f"\n{'='*70}")

if __name__ == '__main__':
    main()

