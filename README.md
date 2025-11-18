# GenDL-LatentControl

> 생성형딥러닝 텀 프로젝트

# 목표

VAE 특징 벡터 추출 테스트  
1) 특징 A를 가진 데이터들을 Encoder에 통과시키고 평균을 냄 (z_1, 혹은 수염이 있는 사람, 모자를 쓴 사람 등)  
2) 특징 A를 가지지 않은 데이터들을 Encoder에 통과시키고 평균을 냄 (z_2)  
3) z_1과 z_2의 차이를 연산하여 특징 A의 벡터를 추출 (v_g = z_1 - z_1로)  
4) 특징 A를 가지지 않은 데이터를 Encoder에 통과시키고, v_g를 더하여 Decoder에 통과시킴 (I_1)  
5) I_1이 특징 A를 가진 해당 데이터로 나오는지 확인  

# 모델

- VAE-CelebA: https://huggingface.co/hussamalafandi/VAE-CelebA

# 데이터셋

- celebA: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

# 셋업

모델(VAE-CelebA)과 데이터(celebA)를 로컬로 내려받아야 합니다. (macOS)
1) 폴더 구조
    - 프로젝트 루트
        - dataset/
        - models/
        - src/
    - 프로젝트 하위 폴더 생성
        ```
        cd /(프로젝트 위치)/GenDL-LatentControl
        mkdir model
        mkdir dataset
        ```
2) Hugging Face에서 모델 받기
    - Git LFS 설치 및 초기화:  
        ```
        brew install git-lfs
        git lfs install
        ```
    - 모델 클론:  
        ```
        cd /(프로젝트 위치)/GenDL-LatentControl/model
        git clone https://huggingface.co/hussamalafandi/VAE-CelebA vae-celeba
        ```
3) Kaggle에서 CelebA 데이터 받기
    - Kaggle CLI 설치 및 인증:
        ```
        pip3 install kaggle
        # kaggle.json을 받아서 ~/.kaggle/kaggle.json에 저장
        chmod 600 ~/.kaggle/kaggle.json
        ```
    - 데이터 다운로드 및 압축 해제:
        ```
        cd /(프로젝트 위치)/GenDL-LatentControl/dataset
        mkdir celebA
        kaggle datasets download -d jessicali9530/celeba-dataset -p celebA --unzip
        ```
