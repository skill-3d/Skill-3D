import base64
import logging
import os

import cv2
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

log_dir = os.environ.get("SKILL3D_LOG_DIR", "/tmp/skill3d_logs")
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, 'sam3_client.log'))
file_handler.setFormatter(
    logging.Formatter('%(name)s - %(levelname)s - %(message)s')
)
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)


class SAM3Client:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')

    def health_check(self):
        try:
            response = requests.get(f'{self.server_url}/health', timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error('Health check failed: %s', exc)
            return None

    def infer(self, image_path: str, text_prompt: str = None, box=None, conf: float = 0.3):
        try:
            if not os.path.exists(image_path):
                logger.error('Image file does not exist: %s', image_path)
                return None

            image = cv2.imread(image_path)
            if image is None:
                logger.error('Failed to read image: %s', image_path)
                return None

            ok, buffer = cv2.imencode('.jpg', image)
            if not ok:
                logger.error('Failed to encode image: %s', image_path)
                return None

            payload = {
                'image': base64.b64encode(buffer.tobytes()).decode('utf-8'),
                'conf': conf,
            }
            if text_prompt:
                payload['text_prompt'] = text_prompt
            if box is not None:
                payload['box'] = box

            response = requests.post(
                f'{self.server_url}/infer',
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()

            if not result.get('success'):
                logger.error('Server error: %s', result)
                return None

            overlay = image.copy()
            mask_array = None
            for idx, mask_info in enumerate(result.get('masks', [])):
                mask_bytes = base64.b64decode(mask_info['mask'])
                decoded = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                if decoded is None:
                    continue
                if mask_array is None:
                    mask_array = np.zeros_like(decoded)
                mask_array = np.maximum(mask_array, decoded)
                color = np.array([
                    (37 * (idx + 1)) % 255,
                    (97 * (idx + 1)) % 255,
                    (177 * (idx + 1)) % 255,
                ], dtype=np.uint8)
                colored = np.zeros_like(image)
                colored[decoded > 0] = color
                overlay[decoded > 0] = cv2.addWeighted(
                    overlay[decoded > 0], 0.55, colored[decoded > 0], 0.45, 0
                )

            os.makedirs('outputs', exist_ok=True)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            overlay_path = f'outputs/sam3_overlay_{stem}.png'
            combined_path = f'outputs/sam3_combined_{stem}.png'
            mask_path = f'outputs/sam3_mask_{stem}.png'

            combined = np.vstack([image, overlay])
            cv2.imwrite(overlay_path, overlay)
            cv2.imwrite(combined_path, combined)
            if mask_array is not None:
                cv2.imwrite(mask_path, mask_array)

            return {
                'success': True,
                'output_path': combined_path,
                'overlay_path': overlay_path,
                'mask_path': mask_path if mask_array is not None else None,
                'shape': result.get('shape'),
                'scores': result.get('scores', []),
                'boxes': result.get('boxes', []),
                'masks': result.get('masks', []),
                'raw': result,
            }
        except Exception as exc:
            logger.error('Inference request failed: %s', exc)
            return None
