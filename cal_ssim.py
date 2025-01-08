import os
import numpy as np
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread

raw_folder = "./results/DualDn_Big_archived_20241210_113216/sRGB"
visuals_folder = "./results/DualDn_Big_archived_20241210_113216/visuals"

ssim_scores = []

for raw_file in os.listdir(raw_folder):
    if raw_file.endswith(".mat"):
        # Raw 폴더의 파일 이름에서 ID 추출
        file_id = raw_file.replace(".mat", "")
        
        # 대응하는 visuals 파일 찾기
        visuals_file = f"{file_id}_ours.png"
        visuals_path = os.path.join(visuals_folder, visuals_file)

        if os.path.exists(visuals_path):
            # .mat 파일 로드
            raw_path = os.path.join(raw_folder, raw_file)
            raw_data = loadmat(raw_path)

            # .mat 파일에서 첫 번째 배열 추출 (키 이름은 상황에 맞게 수정)
            raw_image = None
            for key in raw_data:
                if isinstance(raw_data[key], np.ndarray):
                    raw_image = raw_data[key]
                    break

            if raw_image is None:
                print(f"No valid array found in {raw_file}")
                continue

            # .png 파일 로드
            visuals_image = imread(visuals_path, as_gray=True)

            # 크기 맞추기 (필요 시 추가)
            if raw_image.shape != visuals_image.shape:
                min_shape = np.minimum(raw_image.shape, visuals_image.shape)
                raw_image = raw_image[:min_shape[0], :min_shape[1]]
                visuals_image = visuals_image[:min_shape[0], :min_shape[1]]

            # SSIM 계산
            score = ssim(raw_image, visuals_image, data_range=visuals_image.max() - visuals_image.min())
            ssim_scores.append((file_id, score))
        else:
            print(f"Visuals file not found for {file_id}")

for file_id, score in ssim_scores:
    print(f"SSIM for {file_id}: {score}")

if ssim_scores:
    mean_ssim = np.mean([score for _, score in ssim_scores])
    print(f"Average SSIM: {mean_ssim}")
