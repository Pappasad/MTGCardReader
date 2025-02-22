from paddleocr import PaddleOCR
import paddle
import os
import cv2
import numpy as np
from PIL import Image

# ocr = PaddleOCR(lang="en", use_gpu=True, show_long=False, use_tensorrt=False)
# result = ocr.ocr("test_image.png", cls=True)
# print(result)

class NewOCR:

    def __init__(self, gpu=False): #GPU only works in hinton
        self.reader = PaddleOCR(lang='en', use_angle_cls=True, show_log=False, use_gpu=gpu)

    def __call__(self, img_path: str | Image.Image, visualize=False, returnAll=False):  
        if not isinstance(img_path, str):
            result = self.reader.ocr(np.array(img_path), cls=True)
        else:
            result = self.reader.ocr(img_path, cls=True)

        if visualize:
            directory = os.path.join('data', 'NewOCRTemp')
            os.makedirs(directory, exist_ok=True)
            img = cv2.imread(img_path)
            for line in result:
                for box_info in line:
                    pts, (txt, score) = box_info
                    cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, (255, 0, 0), 2)
                    x, y = int(pts[0][0]), int(pts[0][1]) - 5
                    cv2.putText(img, str(txt), (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)

            cv2.imwrite(os.path.join(directory, os.path.basename(img_path)), img)

        if not returnAll:
            if result[0]:
                return result[0][0][1][0]
            else:
                return ' '
        else:       
            if result[0]: 
                to_ret = np.empty((len(result[0]), 3), dtype=object)
                to_ret[:] = None
                for j, item in enumerate(result[0]):
                    coords, (text, confidence) = item
                    to_ret[j][0] = np.array(coords, dtype=int)
                    to_ret[j][1] = text
                    to_ret[j][2] = confidence
                return to_ret
            else:
                return ' '
        


if __name__ == '__main__':
    ocr = NewOCR(gpu=True)
    print(ocr('test_image.png'))