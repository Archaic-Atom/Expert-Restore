import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 读取图像
image_path = '/data/zzd/Datasets/UIR/WaveUIR/test/derain/Rain100L/input/1.png'  # 替换为你的图像路径
gt_image_path = '/data/zzd/Datasets/UIR/WaveUIR/test/derain/Rain100L/target/1.png'  # 替换为你的图像路径
image = Image.open(image_path).convert('L')  # 转换为灰度图像
image_array = np.array(image)
gt_image = Image.open(gt_image_path).convert('L')  # 转换为灰度图像
gt_image_array = np.array(gt_image)

res_img = np.abs(image_array - gt_image_array)

res_fft_result = np.fft.fft2(res_img)
res_fft_result_shifted = np.fft.fftshift(res_fft_result)  # 将零频率分量移到中心
res_magnitude_spectrum = np.log(np.abs(res_fft_result_shifted) + 1)  # 取对数以便更好地显示

# 进行 FFT
fft_result = np.fft.fft2(image_array)
fft_shifted = np.fft.fftshift(fft_result)  # 将零频率分量移到中心

# 计算幅度谱
magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)  # 取对数以便更好地显示

plt.axis('off') # 去坐标轴
plt.xticks([]) # 去刻度
plt.imshow(res_magnitude_spectrum, cmap='magma')
plt.savefig("./fft.png", bbox_inches='tight', pad_inches=-0.1)

# 进行 IFFT
ifft_result = np.fft.ifft2(fft_result)
reconstructed_image = np.abs(ifft_result)  # 取绝对值得到重建的图像

# # 显示结果
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 3, 1)
# plt.imshow(image_array, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(res_magnitude_spectrum, cmap='gray')
# plt.title('Magnitude Spectrum')
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.imshow(reconstructed_image, cmap='gray')
# plt.title('Reconstructed Image')
# plt.axis('off')

# # plt.show()
# plt.savefig("./fft.png")