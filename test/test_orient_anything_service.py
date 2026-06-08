#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from skill3d.tools.orient_anything_tool import OrientAnythingTool


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Orient-Anything service with a local image.")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the local image file to test.",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:20034",
        help="Orient-Anything service URL.",
    )
    parser.add_argument(
        "--crop-box",
        required=True,
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Target object crop box for the reference image.",
    )
    parser.add_argument(
        "--target-image",
        default=None,
        help="Optional second image for relative pose testing.",
    )
    parser.add_argument(
        "--target-crop-box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Target object crop box for the second image.",
    )
    parser.add_argument(
        "--render-visualization",
        action="store_true",
        help="Request rendered overlay images from the service.",
    )
    parser.add_argument(
        "--remove-background",
        action="store_true",
        help="Enable background removal before inference.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"[ERROR] image not found: {image_path}", file=sys.stderr)
        return 1

    target_image_path = None
    if args.target_image:
        target_image_path = Path(args.target_image).expanduser().resolve()
        if not target_image_path.exists():
            print(f"[ERROR] target image not found: {target_image_path}", file=sys.stderr)
            return 1

    tool = OrientAnythingTool(use_mock=False, server_url=args.server_url)
    print(f"[INFO] Using server: {args.server_url}")

    client = getattr(tool, "_client", None)
    if client is not None and hasattr(client, "health_check"):
        health = client.health_check()
        print("[INFO] Health check:")
        print(json.dumps(health, ensure_ascii=False, indent=2))
    else:
        print("[WARN] No health_check() available on Orient-Anything client")

    result = tool.call(
        image_path=str(image_path),
        crop_box=list(args.crop_box),
        target_image_path=str(target_image_path) if target_image_path else None,
        target_crop_box=list(args.target_crop_box) if args.target_crop_box else None,
        remove_background=args.remove_background,
        render_visualization=args.render_visualization,
    )

    print("[INFO] Inference result:")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    if not result.get("success"):
        return 2

    output_path = result.get("output_path")
    target_output_path = result.get("target_output_path")
    if output_path:
        print(f"[INFO] Reference overlay: {output_path}")
    if target_output_path:
        print(f"[INFO] Target overlay: {target_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
