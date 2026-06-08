import argparse
import base64
import io
import logging
import os
import sys
import traceback
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
)

log_dir = os.environ.get("SKILL3D_LOG_DIR", "/tmp/skill3d_logs")
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, 'sam3_server.log'))
file_handler.setFormatter(
    logging.Formatter('%(name)s - %(levelname)s - %(message)s')
)
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

app = Flask(__name__)

model = None
processor = None
model_name = 'sam3.1'
device_name = None

PROJECT_ROOT = os.environ.get(
    'SKILL3D_PROJECT_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')),
)
DEFAULT_SAM3_REPO = os.path.join(PROJECT_ROOT, 'third_party/sam3')
DEFAULT_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints/sam3.1/sam3.1_multiplex.pt')


def _install_sam3_runtime_shims(sam3_repo: str) -> None:
    if 'pkg_resources' not in sys.modules:
        pkg = types.ModuleType('pkg_resources')

        def resource_filename(package_or_requirement, resource_name):
            if package_or_requirement == 'sam3':
                return os.path.join(sam3_repo, 'sam3', resource_name)
            raise ValueError(package_or_requirement)

        pkg.resource_filename = resource_filename
        sys.modules['pkg_resources'] = pkg

    if 'ftfy' not in sys.modules:
        ftfy = types.ModuleType('ftfy')
        ftfy.fix_text = lambda text: text
        sys.modules['ftfy'] = ftfy

    if 'iopath.common.file_io' not in sys.modules:
        iopath = types.ModuleType('iopath')
        common = types.ModuleType('iopath.common')
        file_io = types.ModuleType('iopath.common.file_io')

        class _PathMgr:
            def open(self, path, mode='r', *args, **kwargs):
                return open(path, mode, *args, **kwargs)

        file_io.g_pathmgr = _PathMgr()
        sys.modules['iopath'] = iopath
        sys.modules['iopath.common'] = common
        sys.modules['iopath.common.file_io'] = file_io

    if sam3_repo not in sys.path:
        sys.path.insert(0, sam3_repo)

    import sam3.model.vitdet as vitdet

    def patched_addmm_act(activation, linear, mat1):
        y = F.linear(
            mat1.float(),
            linear.weight.float(),
            None if linear.bias is None else linear.bias.float(),
        )
        if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            return F.relu(y)
        if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            return F.gelu(y)
        raise ValueError(f'Unexpected activation {activation}')

    vitdet.addmm_act = patched_addmm_act


def _absolute_xyxy_to_normalized_xywh(box_xyxy, width: int, height: int):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = max(0.0, min(x1, width))
    x2 = max(0.0, min(x2, width))
    y1 = max(0.0, min(y1, height))
    y2 = max(0.0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Invalid box: expected [x1, y1, x2, y2] with x2>x1 and y2>y1')
    return [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]


def load_model(checkpoint_path: str, sam3_repo: str, confidence_threshold: float = 0.3):
    global model, processor, device_name
    try:
        logger.info('Loading SAM3.1 model from %s', checkpoint_path)
        _install_sam3_runtime_shims(sam3_repo)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info('Using device: %s', device_name)
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=device_name,
            eval_mode=True,
            compile=False,
        )
        processor = Sam3Processor(
            model,
            device=device_name,
            confidence_threshold=confidence_threshold,
        )
        logger.info('SAM3.1 model loaded successfully')
        return True
    except Exception as exc:
        logger.error('Failed to load SAM3.1 model: %s', exc)
        logger.error(traceback.format_exc())
        model = None
        processor = None
        return False


@app.route('/health', methods=['GET'])
def health_check():
    try:
        status = {
            'status': 'healthy' if model is not None else 'unhealthy',
            'model_name': model_name,
            'device': device_name,
        }
        return jsonify(status), 200 if model is not None else 500
    except Exception as exc:
        return jsonify({'status': 'unhealthy', 'error': str(exc)}), 500


@app.route('/infer', methods=['POST'])
def infer():
    global model, processor
    if model is None or processor is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json(force=True)
        if 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400

        image_bytes = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        width, height = image.size

        text_prompt = data.get('text_prompt') or data.get('prompt')
        box = data.get('box')
        if not text_prompt and not box:
            return jsonify({'error': 'At least one of text_prompt or box is required'}), 400

        confidence_threshold = float(data.get('conf', data.get('confidence_threshold', 0.3)))
        processor.set_confidence_threshold(confidence_threshold)

        state = processor.set_image(image)
        if text_prompt:
            state = processor.set_text_prompt(prompt=text_prompt, state=state)
        if box is not None:
            normalized_xywh = _absolute_xyxy_to_normalized_xywh(box, width=width, height=height)
            state = processor.add_geometric_prompt(box=normalized_xywh, label=True, state=state)

        masks = state['masks'].detach().cpu().numpy()
        boxes = state['boxes'].detach().cpu().numpy().tolist()
        scores = state['scores'].detach().cpu().numpy().tolist()

        if len(scores) == 0:
            return jsonify({
                'success': True,
                'mask': None,
                'masks': [],
                'scores': [],
                'boxes': [],
                'shape': [height, width],
            })

        mask_list = []
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for idx, mask in enumerate(masks):
            mask_bool = np.squeeze(mask).astype(bool)
            mask_uint8 = (mask_bool.astype(np.uint8) * 255)
            combined_mask = np.maximum(combined_mask, mask_uint8)
            ok, buffer = cv2.imencode('.png', mask_uint8)
            if not ok:
                raise RuntimeError('Failed to encode mask PNG')
            mask_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            mask_list.append({
                'id': idx,
                'mask': mask_b64,
                'score': float(scores[idx]),
                'box': boxes[idx],
            })

        ok, buffer = cv2.imencode('.png', combined_mask)
        if not ok:
            raise RuntimeError('Failed to encode combined mask PNG')

        return jsonify({
            'success': True,
            'mask': base64.b64encode(buffer.tobytes()).decode('utf-8'),
            'masks': mask_list,
            'scores': scores,
            'boxes': boxes,
            'shape': [height, width],
            'text_prompt': text_prompt,
        })
    except Exception as exc:
        logger.error('Inference failed: %s', exc)
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Inference failed: {exc}'}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAM3.1 image segmentation server')
    parser.add_argument('--checkpoint_path', type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument('--sam3_repo', type=str, default=DEFAULT_SAM3_REPO)
    parser.add_argument('--port', type=int, default=20040)
    parser.add_argument('--conf', type=float, default=0.3)
    args = parser.parse_args()

    logger.info('Starting SAM3.1 server')
    logger.info('checkpoint_path=%s', args.checkpoint_path)
    logger.info('sam3_repo=%s', args.sam3_repo)
    logger.info('port=%s', args.port)

    if not load_model(
        checkpoint_path=args.checkpoint_path,
        sam3_repo=args.sam3_repo,
        confidence_threshold=args.conf,
    ):
        raise SystemExit(1)

    app.run(host='0.0.0.0', port=args.port, debug=False)
