#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from skill3d.tools.swinir_tool import SwinIRTool


def main() -> int:
    parser = argparse.ArgumentParser(description="Test SwinIR service with a local image.")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the local image file to test.",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:20032",
        help="SwinIR service URL.",
    )
    parser.add_argument(
        "--task",
        default="real_sr",
        choices=["real_sr", "classical_sr", "lightweight_sr", "gray_dn", "color_dn", "jpeg_car", "color_jpeg_car"],
        help="SwinIR task mode.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Upscale factor for SR tasks.",
    )
    parser.add_argument(
        "--crop-box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional crop box for local enhancement.",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=0,
        help="Optional crop margin in pixels.",
    )
    parser.add_argument(
        "--large-model",
        action="store_true",
        help="Use large real_sr model if supported.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"[ERROR] image not found: {image_path}", file=sys.stderr)
        return 1

    tool = SwinIRTool(use_mock=False, server_url=args.server_url)
    print(f"[INFO] Using server: {args.server_url}")

    client = getattr(tool, "_client", None)
    if client is not None and hasattr(client, "health_check"):
        health = client.health_check()
        print("[INFO] Health check:")
        print(json.dumps(health, ensure_ascii=False, indent=2))
    else:
        print("[WARN] No health_check() available on SwinIR client")

    result = tool.call(
        image_path=str(image_path),
        crop_box=list(args.crop_box) if args.crop_box else None,
        crop_margin=args.crop_margin,
        task=args.task,
        scale=args.scale,
        large_model=args.large_model,
    )

    print("[INFO] Inference result:")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    if not result.get("success"):
        return 2

    output_path = result.get("output_path")
    if output_path:
        print(f"[INFO] Enhanced output: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
