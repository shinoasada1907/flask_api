import os
from ultralytics import YOLO
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)
from base64 import b64encode
@app.route('/image', methods=['POST'])
def image():
    try:
        # Đọc ảnh từ request
        image_data = io.BytesIO(request.data)
        image = Image.open(image_data)

        # Lưu ảnh gốc vào thư mục static của Flask
        filename = 'image.jpg'
        filepath = os.path.join(os.getcwd(), filename)
        image.save(filepath)

        # Lấy đường dẫn tuyệt đối đến thư mục chứa file best.pt
        model_path = os.path.join(os.getcwd(), 'best.pt')
        model = YOLO(model_path)

        # Dự đoán và tính toán diện tích đối tượng trên mask và ảnh gốc
        results = model.predict(source=image, conf=0.25)
        if not results[0]:
            print('Không có vết nứt trên hình')
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            anh = plt.imread(filepath)
            ax[0].imshow(anh)
            ax[1].imshow(anh)

            ax[0].set_title('Ảnh gốc')
            ax[1].set_title('Ảnh với mask')

            filename_mask = 'mask.jpg'
            filepath_mask = os.path.join(os.getcwd(), filename_mask)
            plt.savefig(filepath_mask, dpi=100)

            # Trả về tên file và diện tích đối tượng để hiển thị trên giao diện
            with open(filepath_mask, "rb") as image_file:
                encoded_string = b64encode(image_file.read()).decode('utf-8')
            phantram = 0
            # Trả về tên file để hiển thị trên giao diện
            return jsonify(
                {'status': 'Không vết nứt trên hình', 'phantram': phantram, 'mask': encoded_string})
        else:
            print('Có vết nứt trên hình')
            masks = results[0].masks.masks.cpu().numpy()
            total_area = np.sum(masks == 1)
            pixel_size = 1  # Giả sử kích thước của mỗi pixel là 1 đơn vị

            # Chuyển đổi masks sang kiểu numpy.ndarray
            maskssau = masks
            maskssau = np.array(maskssau)

            # Chuyển đổi kiểu dữ liệu của masks thành kiểu bool
            maskssau = maskssau.astype(bool)

            # Khởi tạo một mảng numpy 2 chiều để lưu trữ diện tích đoạn giao nhau
            n_masks = maskssau.shape[0]
            intersection_areas = np.zeros((n_masks, n_masks))

            # Tính toán diện tích đoạn giao nhau giữa các cặp mask
            for i in range(n_masks):
                for j in range(i + 1, n_masks):
                    mask1 = maskssau[i]
                    mask2 = maskssau[j]
                    intersection = np.bitwise_and(mask1, mask2)
                    area_intersection = np.sum(intersection == 1)
                    intersection_areas[i, j] = area_intersection
                    intersection_areas[j, i] = area_intersection

            # Tính tổng diện tích đoạn giao nhau của tất cả các mask
            total_intersection_area = np.sum(intersection_areas)
            object_area = total_area * pixel_size * pixel_size - total_intersection_area / 2
            print('Diện tích của đối tượng trên mask là:', object_area, 'đơn vị')

            # Lưu ảnh với mask vào thư mục static của Flask
            anh = plt.imread(filepath)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(anh)
            ax[1].imshow(anh)
            #
            for mask in masks:
                ax[1].imshow(mask, alpha=0.5, extent=[0, anh.shape[1], anh.shape[0], 0])

            ax[0].set_title('Ảnh gốc')
            ax[1].set_title('Ảnh với mask')

            filename_mask = 'mask.jpg'
            filepath_mask = os.path.join(os.getcwd(), filename_mask)
            plt.savefig(filepath_mask, dpi=100)

            # Trả về tên file và diện tích đối tượng để hiển thị trên giao diện
            with open(filepath_mask, "rb") as image_file:
                encoded_string = b64encode(image_file.read()).decode('utf-8')
            image_area = anh.shape[0] * anh.shape[1] * pixel_size * pixel_size
            phantram = round(object_area / image_area * 100, 2)
            print('Mức độ nứt: ', phantram, '%')
            return jsonify(
                {'status': 'Có vết nứt trên hình', 'phantram': phantram, 'mask': encoded_string})

    except Exception as e:
        return jsonify({'status': str(e)})

if __name__ == '__main__':
    app.run(host='192.168.95.33 ', port=5000)