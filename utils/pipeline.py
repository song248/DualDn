# -----------------------------------------------------------------------------------
# [ECCV2024] DualDn: Dual-domain Denoising via Differentiable ISP 
# [Homepage] https://openimaginglab.github.io/DualDn/
# [Author] Originally Written by Ruikang Li, from MMLab, CUHK.
# [Usage] A robust differentiable ISP convert raw images to sRGB images, can be used in end-to-end training.
# [License] Absolutely open-source and free to use, please cite our paper if possible. :)
# -----------------------------------------------------------------------------------

import torch
import kornia
import numpy as np

def pack_raw(im, camera):
    if camera == 'Sony':
        H, W, C = im.shape
        out = np.concatenate((im[0:H:2, 0:W:2, :],
                                im[0:H:2, 1:W:2, :],
                                im[1:H:2, 1:W:2, :],
                                im[1:H:2, 0:W:2, :]), axis=2)
    elif camera == 'Fuji':
        img_shape = im.shape

        H = (img_shape[0] // 6) * 6
        W = (img_shape[1] // 6) * 6

        out = np.zeros((9, H // 3, W // 3), dtype=np.uint16)

        # 0 R
        out[0, 0::2, 0::2] = im[0:H:6, 0:W:6]
        out[0, 0::2, 1::2] = im[0:H:6, 4:W:6]
        out[0, 1::2, 0::2] = im[3:H:6, 1:W:6]
        out[0, 1::2, 1::2] = im[3:H:6, 3:W:6]

        # 1 G
        out[1, 0::2, 0::2] = im[0:H:6, 2:W:6]
        out[1, 0::2, 1::2] = im[0:H:6, 5:W:6]
        out[1, 1::2, 0::2] = im[3:H:6, 2:W:6]
        out[1, 1::2, 1::2] = im[3:H:6, 5:W:6]

        # 1 B
        out[2, 0::2, 0::2] = im[0:H:6, 1:W:6]
        out[2, 0::2, 1::2] = im[0:H:6, 3:W:6]
        out[2, 1::2, 0::2] = im[3:H:6, 0:W:6]
        out[2, 1::2, 1::2] = im[3:H:6, 4:W:6]

        # 4 R
        out[3, 0::2, 0::2] = im[1:H:6, 2:W:6]
        out[3, 0::2, 1::2] = im[2:H:6, 5:W:6]
        out[3, 1::2, 0::2] = im[5:H:6, 2:W:6]
        out[3, 1::2, 1::2] = im[4:H:6, 5:W:6]

        # 5 B
        out[4, 0::2, 0::2] = im[2:H:6, 2:W:6]
        out[4, 0::2, 1::2] = im[1:H:6, 5:W:6]
        out[4, 1::2, 0::2] = im[4:H:6, 2:W:6]
        out[4, 1::2, 1::2] = im[5:H:6, 5:W:6]

        out[5, :, :] = im[1:H:3, 0:W:3]
        out[6, :, :] = im[1:H:3, 1:W:3]
        out[7, :, :] = im[2:H:3, 0:W:3]
        out[8, :, :] = im[2:H:3, 1:W:3]
    return out


def normalize(image, color_mask, black_level, white_level):
    """
    Normalize the image using black and white levels.
    """
    print(f"Black level: {black_level}, White level: {white_level}")

    # 0차원 텐서일 경우 숫자로 변환
    if black_level.dim() == 0:
        black_level = black_level.item()
    if white_level.dim() == 0:
        white_level = white_level.item()

    # black_level과 white_level이 스칼라인 경우 처리
    if isinstance(black_level, (int, float)) and isinstance(white_level, (int, float)):
        normalized_image = (image - black_level) / (white_level - black_level)
    else:
        # black_level과 white_level이 리스트 또는 배열일 경우
        normalized_image = image.clone()
        for i in range(len(black_level)):
            mask = color_mask == i
            normalized_image[mask] = (image[mask] - black_level[i]) / (white_level[i] - black_level[i])

    return normalized_image


# def white_balance(image, color_mask, wb_matrix):

#     wb_mask = torch.ones(image.shape).to(image.device)
#     wb_matrix = torch.div(wb_matrix, torch.min(wb_matrix, -1).values.unsqueeze(-1))

#     for j in range(wb_matrix.shape[0]):
#         for i in range(wb_matrix.shape[-1]):
#             if torch.any(torch.eq(color_mask, i)): 
#                 wb_mask[j] = wb_mask[j].masked_fill(torch.eq(color_mask[j], i), value = wb_matrix[j][0][0][i])

#     wb_image = torch.mul(image, wb_mask)
#     wb_image = torch.clamp(wb_image, min=1e-8, max=1.0)

#     return wb_image
def white_balance(image, color_mask, wb_matrix):
    print(f"wb_matrix shape: {wb_matrix.shape}")
    print(f"color_mask shape: {color_mask.shape}")

    # wb_matrix를 확장하여 브로드캐스팅 가능하도록 변환
    if wb_matrix.ndim == 2:  # [3, 3] 형태
        wb_matrix = wb_matrix.diagonal().view(1, 3, 1, 1)  # 대각선 값을 추출하여 [1, 3, 1, 1]로 확장

    print(f"wb_matrix expanded shape: {wb_matrix.shape}")

    # image의 차원을 [1, 3, H, W]로 변환
    if image.ndim == 3:  # [H, W, 3] 형태라면
        image = image.permute(2, 0, 1).unsqueeze(0)  # [3, H, W] -> [1, 3, H, W]

    # wb_matrix와 브로드캐스팅
    wb_mask = image * wb_matrix

    # 결과를 [H, W, 3] 형식으로 변환
    wb_mask = wb_mask.squeeze(0).permute(1, 2, 0)  # [1, 3, H, W] -> [H, W, 3]

    return wb_mask

def demosaic(image, color_desc, color_mask, rgb_xyz_matrix, demosaic_type='AHD'):

    mask_r = torch.zeros(image.shape).to(image.device)
    mask_g = torch.zeros(image.shape).to(image.device)
    mask_b = torch.zeros(image.shape).to(image.device)

    for i, col in enumerate(color_desc):
        if col == 'R':
            mask_r = mask_r.masked_fill(torch.eq(color_mask, i), 1)
            image_r = torch.mul(image, mask_r)
        if col == 'G':
            mask_g = mask_g.masked_fill(torch.eq(color_mask, i), 1)
            image_g = torch.mul(image, mask_g)
        if col == 'B':
            mask_b = mask_b.masked_fill(torch.eq(color_mask, i), 1)
            image_b = torch.mul(image, mask_b)

    if demosaic_type == 'nearest':
        image_r = image_r + torch.roll(image_r, shifts=1, dims=-2) + torch.roll(image_r, shifts=1, dims=-1) \
        + torch.roll(image_r, shifts=(1, 1), dims=(-1, -2))
        image_g = image_g + torch.roll(image_g, shifts=1, dims=-1)
        image_b = image_b + torch.roll(image_b, shifts=1, dims=-2) + torch.roll(image_b, shifts=1, dims=-1) \
        + torch.roll(image_b, shifts=(1, 1), dims=(-1, -2))

    elif demosaic_type == 'bilinear':
        r_b_kernel1 = torch.tensor([[1, 0, 1],
                                    [0, 0, 0],
                                    [1, 0, 1]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4.0
        r_b_kernel2 = torch.tensor([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 2.0
        g_kernel = torch.tensor([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4.0
        image_r = image_r + torch.nn.functional.conv2d(image_r, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_r, r_b_kernel2, stride=(1, 1), padding="same")
        image_g = image_g + torch.nn.functional.conv2d(image_g, g_kernel, stride=(1, 1), padding="same")
        image_b = image_b + torch.nn.functional.conv2d(image_b, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_b, r_b_kernel2, stride=(1, 1), padding="same")

    elif demosaic_type == 'Malvar':
        f0 = torch.tensor([[0, 0, -1, 0, 0],
                        [0, 0, 2, 0, 0],
                        [-1, 2, 4, 2, -1],
                        [0, 0, 2, 0, 0],
                        [0, 0, -1, 0, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 8

        f1 = torch.tensor(
            [[0, 0, 1, 0, 0],
            [0, -2, 0, -2, 0],
            [-2, 8, 10, 8, -2],
            [0, -2, 0, -2, 0],
            [0, 0, 1, 0, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 16

        f2 = torch.transpose(f1, dim0=-2, dim1=-1).to(image.device)

        f3 = torch.tensor(
            [[0, 0, -3, 0, 0],
            [0, 4, 0, 4, 0],
            [-3, 0, 12, 0, -3],
            [0, 4, 0, 4, 0],
            [0, 0, -3, 0, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 16

        d0 = torch.nn.functional.conv2d(image, f0, stride=1, padding="same")
        d1 = torch.nn.functional.conv2d(image, f1, stride=1, padding="same")
        d2 = torch.nn.functional.conv2d(image, f2, stride=1, padding="same")
        d3 = torch.nn.functional.conv2d(image, f3, stride=1, padding="same")

        mask_r_g_r_row = torch.roll(mask_r, shifts=1, dims=-1)
        mask_r_g_r_col = torch.roll(mask_r, shifts=1, dims=-2)
        image_r = image_r + torch.mul(d1, mask_r_g_r_row) + torch.mul(d2, mask_r_g_r_col) + torch.mul(d3, mask_b)

        image_g = image_g + torch.mul(d0, mask_r) + torch.mul(d0, mask_b)

        mask_b_g_b_row = torch.roll(mask_b, shifts=1, dims=-1)
        mask_b_g_b_col = torch.roll(mask_b, shifts=1, dims=-2)
        image_b = image_b + torch.mul(d1, mask_b_g_b_row) + torch.mul(d2, mask_b_g_b_col) + torch.mul(d3, mask_r)

    elif demosaic_type == 'AHD':
        h0_row = torch.tensor([-1, 2, 2, 2, -1], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) / 4

        h0_col = torch.tensor([[-1],
                        [2],
                        [2],
                        [2],
                        [-1]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4

        d0_row = torch.nn.functional.conv2d(image, h0_row, stride=1, padding="same")
        d0_col = torch.nn.functional.conv2d(image, h0_col, stride=1, padding="same")

        g_row = image_g + torch.mul(d0_row, mask_r) + torch.mul(d0_row, mask_b)
        g_col = image_g + torch.mul(d0_col, mask_r) + torch.mul(d0_col, mask_b)

        r_b_kernel1 = torch.tensor([[1, 0, 1],
                                    [0, 0, 0],
                                    [1, 0, 1]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4.0
        r_b_kernel2 = torch.tensor([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 2.0
        g_kernel = torch.tensor([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4.0

        homo_row1 = torch.tensor([1, 2, -3, 0, 0], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) 
        homo_col1 = torch.tensor([[1],
                        [2],
                        [-3],
                        [0],
                        [0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) 

        homo_row2 = torch.tensor([0, 0, -3, 2, 1], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) 
        homo_col2 = torch.tensor([[0],
                        [0],
                        [-3],
                        [2],
                        [1]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) 

        r_sample = image_r + torch.nn.functional.conv2d(image_r, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_r, r_b_kernel2, stride=(1, 1), padding="same")
        g_sample = image_g + torch.nn.functional.conv2d(image_g, g_kernel, stride=(1, 1), padding="same")
        b_sample = image_b + torch.nn.functional.conv2d(image_b, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_b, r_b_kernel2, stride=(1, 1), padding="same")

        r_row = kornia.filters.gaussian_blur2d((r_sample - g_sample), (3,3), (1.5,1.5)) + g_row
        r_col = kornia.filters.gaussian_blur2d((r_sample - g_sample), (3,3), (1.5,1.5)) + g_col

        b_row = kornia.filters.gaussian_blur2d((b_sample - g_sample), (3,3), (1.5,1.5)) + g_row
        b_col = kornia.filters.gaussian_blur2d((b_sample - g_sample), (3,3), (1.5,1.5)) + g_col

        image_sample = torch.concat([r_sample, g_sample, b_sample], dim=1)

        image_xyz = torch.matmul(rgb_xyz_matrix.unsqueeze(-3), image_sample.permute(0,2,3,1).unsqueeze(-1)).squeeze(-1).permute(0,3,1,2)
        xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)[..., :, None, None].to(image.device).unsqueeze(0)
        xyz_normalized = torch.div(image_xyz, xyz_ref_white)

        threshold = 0.008856
        power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
        scale = 7.787 * xyz_normalized + 4.0 / 29.0
        xyz_int = torch.where(xyz_normalized > threshold, power, scale)

        L = ((116.0 * xyz_int[..., 1, :, :]) - 16.0).unsqueeze(-3)
        a = (500.0 * (xyz_int[..., 0, :, :] - xyz_int[..., 1, :, :])).unsqueeze(-3)
        b = (200.0 * (xyz_int[..., 1, :, :] - xyz_int[..., 2, :, :])).unsqueeze(-3)

        row1 = torch.abs(torch.nn.functional.conv2d(L, homo_row1, stride=(1, 1), padding="same") ) + torch.sqrt(torch.square(torch.nn.functional.conv2d(a, homo_row1, stride=(1, 1), padding="same")) + torch.square(torch.nn.functional.conv2d(b, homo_row1, stride=(1, 1), padding="same")))
        col1 = torch.abs(torch.nn.functional.conv2d(L, homo_col1, stride=(1, 1), padding="same") ) + torch.sqrt(torch.square(torch.nn.functional.conv2d(a, homo_col1, stride=(1, 1), padding="same")) + torch.square(torch.nn.functional.conv2d(b, homo_col1, stride=(1, 1), padding="same")))

        row2 = torch.abs(torch.nn.functional.conv2d(L, homo_row2, stride=(1, 1), padding="same") ) + torch.sqrt(torch.square(torch.nn.functional.conv2d(a, homo_row2, stride=(1, 1), padding="same")) + torch.square(torch.nn.functional.conv2d(b, homo_row2, stride=(1, 1), padding="same")))
        col2 = torch.abs(torch.nn.functional.conv2d(L, homo_col2, stride=(1, 1), padding="same") ) + torch.sqrt(torch.square(torch.nn.functional.conv2d(a, homo_col2, stride=(1, 1), padding="same")) + torch.square(torch.nn.functional.conv2d(b, homo_col2, stride=(1, 1), padding="same")))

        row = row1 + row2
        col = col1 + col2

        image_r = torch.where(row >= col, r_col, r_row)
        image_g = torch.where(row >= col, g_col, g_row)
        image_b = torch.where(row >= col, b_col, b_row)

    demosaic_image = torch.concat([image_r, image_g, image_b], dim=1)

    return demosaic_image


# def rgb_to_srgb(image, rgb_xyz_matrix, ref='D65'):

#     if ref == 'D65':
#         xyz_srgb_matrix = torch.tensor([[3.2404542, -1.5371385, -0.4985314],
#                                         [-0.9692660, 1.8760108, 0.0415560],
#                                         [0.0556434, -0.2040259, 1.0572252]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image.device)
#     elif ref == 'D50':
#         xyz_srgb_matrix = torch.tensor([[3.1338561, -1.6168667, -0.4906146],
#                                         [-0.9787684, 1.9161415, 0.0334540],
#                                         [0.0719453, -0.2289914, 1.4052427]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image.device)

#     xyz_srgb_matrix = xyz_srgb_matrix / torch.sum(xyz_srgb_matrix, axis=-1, keepdims=True)

#     xyz_image = torch.matmul(rgb_xyz_matrix.unsqueeze(-3), image.permute(0,2,3,1).unsqueeze(-1)).squeeze(-1).permute(0,3,1,2)
#     srgb_image = torch.matmul(xyz_srgb_matrix.unsqueeze(-3), xyz_image.permute(0,2,3,1).unsqueeze(-1)).squeeze(-1).permute(0,3,1,2)

#     return srgb_image

def rgb_to_srgb(rgb_image, rgb_xyz_matrix, ref):
    """
    Convert an RGB image to sRGB using the given transformation matrices.
    """
    print(f"rgb_xyz_matrix shape: {rgb_xyz_matrix.shape}")
    print(f"ref shape: {ref.shape}")

    # xyz_srgb_matrix 초기화
    if 'xyz_srgb_matrix' in locals():
        print("xyz_srgb_matrix is already initialized.")
    else:
        xyz_srgb_matrix = torch.eye(3).to(rgb_image.device)  # 기본 단위 행렬로 초기화
        print(f"xyz_srgb_matrix initialized with identity matrix.")

    # Transformation
    xyz_image = rgb_image @ rgb_xyz_matrix.transpose(0, 1)  # RGB -> XYZ 변환
    xyz_srgb_matrix = xyz_srgb_matrix / torch.sum(xyz_srgb_matrix, axis=-1, keepdims=True)  # 정규화

    srgb_image = xyz_image @ xyz_srgb_matrix.transpose(0, 1)  # XYZ -> sRGB 변환
    return srgb_image

# def gamma(image, gamma_type='Rec709'):

#     if gamma_type == 'Rec709':
#         gamma_image1 = 4.5*image.masked_fill(image>=0.018, 0)
#         mask = torch.zeros(image.shape).to(image.device)
#         gamma_image2 = 1.099*torch.pow(image.masked_fill(image<0.018, 0), 0.45) + mask.masked_fill(image>=0.018, -0.099)
#         gamma_image = gamma_image1 + gamma_image2

#     if gamma_type == '2.2':
#         gamma_image = torch.pow(torch.clamp(image, min=1e-8), 1/2.2)

#     return gamma_image

def gamma(image, gamma_type):
    """
    Apply gamma correction based on the gamma type.
    """
    print(f"Gamma type: {gamma_type}")

    if gamma_type == 0:  # Linear
        gamma_image = image
    elif gamma_type == 1:  # sRGB Gamma
        gamma_image = torch.where(image <= 0.0031308,
                                  12.92 * image,
                                  1.055 * torch.pow(image, 1 / 2.4) - 0.055)
    else:
        print(f"Warning: Unknown gamma type {gamma_type}. Defaulting to linear.")
        gamma_image = image  # Default to linear

    return gamma_image


def tone_mapping(image, alpha):

    mid_image = image + alpha*(image)*(1-image)
    tone_mapping_image = mid_image + alpha*(mid_image)*(1-mid_image)

    return tone_mapping_image

def tone_mapping_to_srgb(image, metadata):
    """
    Tone mapping 후 sRGB로 변환
    """
    print("Applying tone mapping and converting to sRGB...")
    # Reinhard 톤 매핑
    tone_mapped_image = image / (1.0 + image)

    # sRGB 감마 보정
    srgb_image = torch.where(
        tone_mapped_image <= 0.0031308,
        12.92 * tone_mapped_image,
        1.055 * torch.pow(tone_mapped_image, 1 / 2.4) - 0.055
    )

    # 클리핑 (sRGB 범위 [0, 1])
    srgb_image = torch.clamp(srgb_image, 0.0, 1.0)

    return srgb_image
def raw_to_srgb_function(image, metadata):
    """
    RAW 이미지를 sRGB로 변환
    """
    # 1. 화이트 밸런스 적용
    wb_matrix = metadata['wb_matrix']
    balanced_image = image * wb_matrix.view(1, 3, 1, 1)  # 채널별 화이트 밸런스 적용

    # 2. 색 공간 변환 (RAW -> XYZ -> sRGB)
    rgb_xyz_matrix = metadata['rgb_xyz_matrix']
    xyz_image = torch.einsum('ij,chw->cihw', rgb_xyz_matrix, balanced_image)  # RGB -> XYZ 변환

    # sRGB 변환 (XYZ -> sRGB)
    xyz_srgb_matrix = torch.tensor([[3.2406, -1.5372, -0.4986],
                                    [-0.9689,  1.8758,  0.0415],
                                    [0.0557, -0.2040,  1.0570]], dtype=torch.float32).to(image.device)
    srgb_image = torch.einsum('ij,cihw->chw', xyz_srgb_matrix, xyz_image)

    # 3. 감마 보정
    srgb_image = torch.where(
        srgb_image <= 0.0031308,
        12.92 * srgb_image,
        1.055 * torch.pow(srgb_image, 1 / 2.4) - 0.055
    )

    # 클리핑 (sRGB 범위 [0, 1] 유지)
    srgb_image = torch.clamp(srgb_image, 0.0, 1.0)

    return srgb_image

def run_pipeline(image, metadata, input_stage, output_stage):
    print(f"Input stage: {input_stage}, Output stage: {output_stage}")

    # 초기 입력 이미지 설정
    current_image = image  # 'image'는 외부에서 전달된 RAW 이미지입니다
    current_stage = 'raw'  # 초기 스테이지를 'raw'로 설정

    # 1. 톤 매핑에서 sRGB로 변환 경로 추가
    if input_stage == 'tone_mapping' and output_stage == 'srgb':
        print("Processing tone_mapping -> sRGB")
        current_image = tone_mapping_to_srgb(current_image, metadata)
        print(f"tone_mapping -> sRGB completed. Image shape: {current_image.shape}")
        return current_image

    # 2. Normalize 단계
    if input_stage == current_stage:
        current_image = normalize(current_image, metadata['color_mask'], metadata['black_level'], metadata['white_level'])
        input_stage = 'normal'
    current_stage = 'normal'
    if current_stage == output_stage:
        return current_image

    # 3. White Balance 단계
    if input_stage == current_stage:
        current_image = white_balance(current_image, metadata['color_mask'], metadata['wb_matrix'])
        input_stage = 'white_balance'
    current_stage = 'white_balance'
    if current_stage == output_stage:
        return current_image

    # 4. Demosaic 단계
    if input_stage == current_stage:
        current_image = demosaic(current_image, metadata['color_desc'], metadata['color_mask'], metadata['rgb_xyz_matrix'], metadata['demosaic_type'])
        input_stage = 'demosaic'
    current_stage = 'demosaic'
    if current_stage == output_stage:
        return current_image

    # 5. RGB to sRGB 단계
    if input_stage == current_stage:
        current_image = rgb_to_srgb(current_image, metadata['rgb_xyz_matrix'], metadata['ref'])
        input_stage = 'rgb_to_srgb'
    current_stage = 'rgb_to_srgb'
    if current_stage == output_stage:
        return current_image

    # 6. Gamma 보정 단계
    if input_stage == current_stage:
        current_image = gamma(current_image, metadata['gamma_type'])
        input_stage = 'gamma'
    current_stage = 'gamma'
    if current_stage == output_stage:
        return current_image

    # 7. Tone Mapping 단계
    if input_stage == current_stage:
        current_image = tone_mapping(current_image, metadata['alpha'])
        input_stage = 'tone_mapping'
    current_stage = 'tone_mapping'
    if current_stage == output_stage:
        return current_image

    # 지원되지 않는 입력/출력 스테이지 조합
    raise ValueError(f"Invalid input/output stage: input_stage = {input_stage}, output_stage = {output_stage}")
