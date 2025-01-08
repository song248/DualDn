from scipy.io import savemat
from PIL import Image
import numpy as np

def image_to_mat(image_path, mat_path):
    # 이미지를 로드하고 numpy 배열로 변환
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)  # HxWxC 배열
    # 필요한 경우 전처리 추가 (예: 정규화)

    # .mat 파일로 저장
    savemat(mat_path, {'image': image_array})

# 사용 예시
image_to_mat('./test_sample_1_Real.png', './test_sample_1_Real.mat')
