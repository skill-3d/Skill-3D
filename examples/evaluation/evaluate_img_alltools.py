import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from tqdm import tqdm
import cv2
import numpy as np
import argparse
from datetime import datetime
from collections import Counter

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from skill3d import SPAgent
from skill3d.core import DataCollector
from skill3d.models import GPTModel, QwenModel
from skill3d.tools import (
    DepthEstimationTool,
    SegmentationTool,
    ObjectDetectionTool,
    SupervisionTool,  # 辅助工具，通常不需要独立实例化
    YOLOETool,        # 如果需要 YOLO 可以启用
    MoondreamTool,    # 轻量级 VLM 工具
    Pi3Tool,          # 主动视觉/3D 工具
    SwinIRTool,       # 图像增强工具
    OrientAnythingTool,  # 朝向/相对旋转工具
)
from skill3d.utils.utils import (
    load_json_data, 
    print_evaluation_results, 
)
# 假设这些评估函数都在外部模块中
from skill3d_evaluation import evaluate_tool_config


def _env_flag(name: str, default: str = "0") -> bool:
    value = str(os.environ.get(name, default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


# Define server URLs
# 注意：请确保这些端口对应的服务已启动
# 如果是本地服务，请将 IP 改为 127.0.0.1
TOOL_SERVERS = {
    "depth": os.environ.get("DEPTH_SERVER_URL", "http://127.0.0.1:20019"),       # Depth Anything V3
    "segmentation": "http://127.0.0.1:20020", # SAM (Segment Anything)
    "detection": "http://127.0.0.1:20022",    # Grounding DINO / YOLOWorld
    "pi3": "http://127.0.0.1:20030",          # Pi3 Active Vision
    "moondream": "http://127.0.0.1:20024",    # Moondream VQA
    "swinir": "http://127.0.0.1:20032",       # SwinIR image restoration
    "orient_anything": os.environ.get("ORIENT_ANYTHING_SERVER_URL", "http://127.0.0.1:20034"),  # Orient-Anything
}

# ==========================================
# 默认工具集：Moondream 默认不启用，需要时通过环境变量显式开启。
# ==========================================
FULL_STACK_TOOLS = [
    # 1. 主动视觉工具 (移动相机)
    Pi3Tool(use_mock=False, server_url=TOOL_SERVERS["pi3"], mode='inference'),

    # 2. 深度估计工具 (判断距离/前后关系)
    DepthEstimationTool(use_mock=False, server_url=TOOL_SERVERS["depth"]),

    # 3. 目标检测工具 (定位物体/画框)
    ObjectDetectionTool(use_mock=False, server_url=TOOL_SERVERS["detection"]),

    # 4. 图像分割工具 (获取物体掩码)
    SegmentationTool(use_mock=False, server_url=TOOL_SERVERS["segmentation"]),

    # 5. 图像增强工具 (低清/模糊图先增强)
    SwinIRTool(use_mock=False, server_url=TOOL_SERVERS["swinir"]),

    # 6. 物体朝向与相对旋转工具
    OrientAnythingTool(use_mock=False, server_url=TOOL_SERVERS["orient_anything"]),
]

if _env_flag("SPAGENT_ENABLE_MOONDREAM_TOOL", "0") or _env_flag("SPAGENT_USE_REAL_MOONDREAM", "0"):
    FULL_STACK_TOOLS.insert(
        4,
        MoondreamTool(use_mock=False, server_url=TOOL_SERVERS["moondream"]),
    )

TOOL_CONFIGS = {
    "full_stack_tools": FULL_STACK_TOOLS
}

def print_pi3_statistics(results: Dict[str, Any]):
    """
    打印 Pi3 工具的参数统计信息 (保持原样)
    """
    if "pi3_statistics" not in results:
        print("\nNo Pi3 tool calls detected in this evaluation.")
        return
    
    pi3_stats = results["pi3_statistics"]
    
    print(f"\n{'='*80}")
    print("Pi3 Tool Parameter Statistics")
    print(f"{'='*80}")
    print(f"Total Pi3 calls:              {pi3_stats['total_pi3_calls']}")
    print(f"Unique angle combinations:    {pi3_stats['unique_angle_combinations']}")
    
    # Angle distribution
    print(f"\n{'Top 5 Most Used Angle Combinations:':<50}")
    print(f"{'Angle (azimuth, elevation)':<35} {'Count':<10} {'Percentage':<15}")
    print("-" * 80)
    
    for item in pi3_stats.get('top_5_angle_combinations', []):
        angle = item['angle']
        count = item['count']
        percentage = item['percentage']
        print(f"{angle:<35} {count:<10} {percentage:<15}")
    
    # rotation_reference_camera usage
    print(f"\n{'Rotation Reference Camera Usage:':<50}")
    print(f"{'Camera':<20} {'Count':<10} {'Percentage':<15}")
    print("-" * 80)
    
    if 'rotation_reference_camera_usage' in pi3_stats:
        for camera, count in pi3_stats['rotation_reference_camera_usage'].items():
            percentage = pi3_stats['rotation_reference_camera_percentage'][camera]
            print(f"{camera:<20} {count:<10} {percentage:<15}")
    
    # camera_view usage
    print(f"\n{'Camera View Mode Usage:':<50}")
    print(f"{'Mode':<20} {'Count':<10} {'Percentage':<15}")
    print("-" * 80)
    
    if 'camera_view_usage' in pi3_stats:
        for mode, count in pi3_stats['camera_view_usage'].items():
            percentage = pi3_stats['camera_view_percentage'][mode]
            display_mode = "Enabled (True)" if "true" in str(mode).lower() else "Disabled (False)"
            print(f"{display_mode:<20} {count:<10} {percentage:<15}")
    
    print(f"{'='*80}\n")

def main():
    """Main function"""
    # Configure paths
    parser = argparse.ArgumentParser(description='Skill-3D Evaluation Script (Full Stack)')

    parser.add_argument('--data_path', type=str, default='dataset/cvbench_data.jsonl',
                        help='Path to the data file')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (default: None for all)')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads')
    parser.add_argument('--image_base_path', type=str, default='dataset',
                        help='Path to the image base directory')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Model to use for evaluation')
    parser.add_argument('--model_backend', type=str, default='gpt',
                        choices=['gpt', 'qwen', 'qwen_vllm'],
                        help='Model backend: gpt for OpenAI-compatible GPT endpoint, qwen_vllm for local vLLM Qwen.')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='Maximum number of tool-call iterations')
    parser.add_argument('--task', type=str, default="all",
                        help='Task to evaluate')
    parser.add_argument('--prompt_style', type=str, default="normal", choices=["normal", "skills"],
                        help='Prompt style to use: normal or skills (default: normal)')
    parser.add_argument('--memory_mode', type=str, default="symbolic", choices=["none", "symbolic", "dense", "hybrid"],
                        help='Skill memory mode: none, symbolic, dense, or hybrid')
    parser.add_argument('--retrieval_mode', type=str, default=None, choices=["symbolic", "dense", "hybrid"],
                        help='Retrieval mode override. Defaults to memory_mode when applicable.')
    parser.add_argument('--retrieval_top_k', type=int, default=6,
                        help='Number of retrieved skills to inject when dense/hybrid retrieval is enabled.')
    parser.add_argument('--retrieval_trace_path', type=str, default=None,
                        help='Optional JSONL path for per-sample retrieval traces.')
    parser.add_argument('--skill_storage_path', type=str,
                        default=os.environ.get("SKILL3D_SKILL_STORAGE_PATH") or os.environ.get("SPAGENT_SKILL_STORAGE_PATH"),
                        help='Benchmark-specific skill storage path. Use separate paths to prevent dynamic skill sharing.')
    
    # Data collection arguments (Script 1 的核心特性)
    parser.add_argument('--enable_data_collection', action='store_true',
                        help='Enable training data collection')
    parser.add_argument('--data_output_dir', type=str, default=None,
                        help='Directory for training data (auto-generated if not specified)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from rollout checkpoint and skip completed sample ids.')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to rollout checkpoint JSONL. Defaults to data_output_dir/rollout_checkpoint.jsonl when data collection is enabled.')
    parser.add_argument('--resume_retry_failed', action='store_true',
                        help='When resuming, retry samples whose checkpointed result was failed.')
    parser.add_argument('--disable_skill_update', action='store_true',
                        help='Disable delayed GT/reward feedback updates. Use this for test-set Skill-Guided Inference.')

    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        # 这里为了防止直接报错退出，可以在调试时注释掉 return
        # return

    if not os.path.exists(args.image_base_path):
        print(f"Error: Image base path not found at {args.image_base_path}")
        # return

    # Run evaluation for each tool configuration
    all_results = {}
    
    for config_name, tools in TOOL_CONFIGS.items():
        # Setup DataCollector if enabled
        data_collector = None
        if args.enable_data_collection:
            if args.data_output_dir is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                data_output_dir = f"training_data/{config_name}_{args.model.replace('-', '_')}_{timestamp}"
            else:
                data_output_dir = args.data_output_dir
            
            data_collector = DataCollector(
                output_dir=data_output_dir,
                save_images=True,  # 保存中间过程生成的图片
                auto_save=True
            )
            print(f"✓ Data collection enabled: {data_output_dir}")
        
        # 调用核心评估函数 (来自 skill3d_evaluation)
        results = evaluate_tool_config(
            config_name=config_name,
            tools=tools,
            data_path=args.data_path,
            image_base_path=args.image_base_path,
            model=args.model,
            model_backend=args.model_backend,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            max_iterations=args.max_iterations,
            data_collector=data_collector,
            prompt_style=args.prompt_style,
            memory_mode=args.memory_mode,
            retrieval_mode=args.retrieval_mode or args.memory_mode,
            retrieval_top_k=args.retrieval_top_k,
            retrieval_trace_path=args.retrieval_trace_path,
            skill_storage_path=args.skill_storage_path,
            checkpoint_path=args.checkpoint_path,
            resume=args.resume,
            resume_retry_failed=args.resume_retry_failed,
            enable_skill_update=not args.disable_skill_update,
        )
        all_results[config_name] = results
        
        # Print individual config results
        print(f"\nResults for {config_name}:")
        print_evaluation_results(results)
        
        # Print Pi3 parameter statistics
        # 即使使用了其他 CV 工具，如果调用了 Pi3，这里依然会统计
        print_pi3_statistics(results)
    
    # Save all results to file
    output_file = f"skill3d_evaluation_results_{args.model.replace('-', '_')}_{args.max_iterations}_{args.task}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {output_file}")

if __name__ == "__main__":
    main()
