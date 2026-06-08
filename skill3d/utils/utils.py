from typing import List, Dict, Any, Tuple, Optional
import json
import os
import re
import pandas as pd


VSI_BENCH_NUMERICAL_TASKS = {
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
    "object_abs_distance",
}

VSI_BENCH_MRA_THRESHOLDS = [round(0.50 + 0.05 * idx, 2) for idx in range(10)]
VSI_BENCH_RELAXED_MRA_THRESHOLDS = [round(0.30 + 0.05 * idx, 2) for idx in range(14)]


def load_json_data(data_path: str) -> List[Dict[str, Any]]:
    """加载json数据集
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        数据列表
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def validate_sample_paths(
    sample: Dict[str, Any],
    base_path: str,
    required_field: str = "image"
) -> Tuple[bool, Dict[str, Any]]:
    """验证样本路径和对话
    
    Args:
        sample: 数据样本
        base_path: 基础路径
        required_field: 需要验证的字段名称（"image" 或 "video"）
        
    Returns:
        (是否有效, 错误信息字典)
    """
    # 提取路径
    paths = sample.get(required_field, [])
    if not paths:
        return False, {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": f"No {required_field} found"
        }
    
    # 验证所有路径是否存在
    full_paths = []
    missing_paths = []
    
    for path in paths:
        full_path = os.path.join(base_path, path)
        full_paths.append(full_path)
        
        if not os.path.exists(full_path):
            missing_paths.append(full_path)
    
    # 如果有路径不存在，返回错误
    if missing_paths:
        return False, {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": f"{required_field.capitalize()} not found: {missing_paths}"
        }
    
    # 提取问题和答案
    conversation = sample.get("conversations", [])
    if not conversation:
        return False, {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": "No conversation found"
        }
    
    question, ground_truth = extract_question_and_answer(conversation)
    if not question or not ground_truth:
        return False, {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": "Question or answer not found"
        }
    
    # 返回验证成功和路径信息
    return True, {
        "path": full_paths,
        "question": question,
        "ground_truth": ground_truth
    }

def extract_question_and_answer(conversation: List[Dict[str, str]]) -> Tuple[str, str]:
    """从对话中提取问题和答案
    
    Args:
        conversation: 对话列表
        
    Returns:
        (问题, 答案) 元组
    """
    # 找到人类的问题
    human_message = None
    for msg in conversation:
        if msg["from"] == "human":
            human_message = msg["value"]
            break
    
    # 找到GPT的答案
    gpt_answer = None
    for msg in conversation:
        if msg["from"] == "gpt":
            gpt_answer = msg["value"]
            break
    
    return human_message, gpt_answer

def normalize_answer(answer: str) -> tuple[str, str]:
    """Normalize answer format
    
    Args:
        answer: Original answer string
        
    Returns:
        Tuple (analysis, final_answer): Analysis content and normalized answer
    """
    original_answer = answer.strip()
    
    # Extract answer part
    processed_answer = original_answer
    answer_start = processed_answer.find("<answer>")
    answer_end = processed_answer.find("</answer>")
    
    if answer_start != -1 and answer_end != -1 and answer_end > answer_start:
        processed_answer = processed_answer[answer_start+8:answer_end].strip()
    
    # Extract option letter if present. Some benchmarks, e.g. CV-Bench Count,
    # have more than four choices, so accept A-Z rather than only A-D.
    final_answer = ""
    option_patterns = [
        r'\(([A-Z])\)',
        r'\b(?:answer|option|choice|choose|select|prediction|final answer|答案)\b\s*(?:is|are|:|=|-|为|是)?\s*\(?\s*([A-Z])\s*\)?\b',
        r'^\s*([A-Z])\s*(?:[.)、:：]|\s*$)',
        r'(?m)^\s*([A-Z])\s*$',
    ]
    for pattern in option_patterns:
        match = re.search(pattern, processed_answer, flags=re.IGNORECASE)
        if match:
            final_answer = match.group(1).upper()
            break
    
    # If no option letter found, return the processed answer
    if not final_answer:
        final_answer = processed_answer
    
    # Since we no longer have <analysis> tags, return empty string for analysis
    return "", final_answer


def extract_numeric_value(answer: str) -> Optional[float]:
    """Extract a scalar numeric value from free-form answer text.

    For simple numeric ranges like ``2-3 meters`` or ``2 to 3 meters``,
    return the midpoint to match benchmark-style tolerant numeric scoring.
    """
    if answer is None:
        return None

    text = str(answer).strip()
    if not text:
        return None

    text = text.replace(",", "")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("m²", " square meters ")
    text = re.sub(r"\*\*", "", text)

    range_match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        left = float(range_match.group(1))
        right = float(range_match.group(2))
        return (left + right) / 2.0

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None

    return float(matches[0])


def compute_vsi_bench_mra(
    predicted_value: Optional[float],
    ground_truth_value: Optional[float],
    thresholds: Optional[List[float]] = None,
) -> float:
    """Compute the VSI-Bench Mean Relative Accuracy (MRA).

    The paper defines MRA as the average correctness under multiple
    relative-error thresholds. For each threshold ``theta``:

    ``correct(theta) = 1[ |pred-gt| / gt <= 1-theta ]``

    averaged over ``theta in {0.50, 0.55, ..., 0.95}``.
    """
    if thresholds is None:
        thresholds = VSI_BENCH_MRA_THRESHOLDS

    if predicted_value is None or ground_truth_value is None:
        return 0.0

    gt = float(ground_truth_value)
    pred = float(predicted_value)

    if gt == 0.0:
        return 1.0 if pred == 0.0 else 0.0

    relative_error = abs(pred - gt) / abs(gt)
    total = 0.0
    for theta in thresholds:
        total += 1.0 if relative_error <= (1.0 - float(theta)) else 0.0
    return total / len(thresholds)


def vsi_bench_metric_name(task: str) -> str:
    return "MRA" if str(task or "").strip() in VSI_BENCH_NUMERICAL_TASKS else "ACC"


def vsi_bench_relaxed_metric_name(task: str) -> str:
    return "relaxed_MRA" if str(task or "").strip() in VSI_BENCH_NUMERICAL_TASKS else "ACC"


def score_vsi_bench_prediction(
    task: str,
    prediction: str,
    ground_truth: str,
) -> Dict[str, Any]:
    """Score a VSI-Bench prediction with both paper and relaxed metrics."""
    _, normalized_prediction = normalize_answer(prediction)
    _, normalized_ground_truth = normalize_answer(ground_truth)

    metric_name = vsi_bench_metric_name(task)
    relaxed_metric_name = vsi_bench_relaxed_metric_name(task)
    parsed_prediction = extract_numeric_value(normalized_prediction) if metric_name == "MRA" else None
    parsed_ground_truth = extract_numeric_value(normalized_ground_truth) if metric_name == "MRA" else None

    if metric_name == "MRA":
        paper_metric_score = compute_vsi_bench_mra(parsed_prediction, parsed_ground_truth)
        relaxed_metric_score = compute_vsi_bench_mra(
            parsed_prediction,
            parsed_ground_truth,
            thresholds=VSI_BENCH_RELAXED_MRA_THRESHOLDS,
        )
        metric_score = paper_metric_score
        is_correct = paper_metric_score >= 0.999999
        is_relaxed_correct = relaxed_metric_score >= 0.999999
    else:
        metric_score = 1.0 if normalized_prediction == normalized_ground_truth else 0.0
        paper_metric_score = metric_score
        relaxed_metric_score = metric_score
        is_correct = bool(metric_score)
        is_relaxed_correct = is_correct

    return {
        "metric_name": metric_name,
        "metric_score": float(metric_score),
        "paper_metric_name": metric_name,
        "paper_metric_score": float(paper_metric_score),
        "relaxed_metric_name": relaxed_metric_name,
        "relaxed_metric_score": float(relaxed_metric_score),
        "is_correct": bool(is_correct),
        "is_relaxed_correct": bool(is_relaxed_correct),
        "normalized_prediction": normalized_prediction,
        "normalized_ground_truth": normalized_ground_truth,
        "parsed_prediction": parsed_prediction,
        "parsed_ground_truth": parsed_ground_truth,
    }


def print_evaluation_results(results: Dict[str, Any]):
    """打印评估结果
    
    Args:
        results: 评估结果字典
    """
    print("\n" + "="*60)
    print("BLINK DATASET EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {results['model']}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Successful samples: {results['successful_samples']}")
    print(f"Failed samples: {results['failed_samples']}")
    overall_metric_label = results.get("overall_metric_label", "Score")
    overall_score = results.get("overall_score", results.get("overall_accuracy", 0.0))
    print(f"Overall {overall_metric_label.lower()}: {overall_score:.4f} ({overall_score*100:.2f}%)")
    if "overall_relaxed_score" in results:
        relaxed_label = results.get("overall_relaxed_metric_label", "Relaxed score")
        relaxed_score = results.get("overall_relaxed_score", 0.0)
        print(f"Overall {relaxed_label.lower()}: {relaxed_score:.4f} ({relaxed_score*100:.2f}%)")
    print(f"Average inference time: {results['average_inference_time']:.2f} seconds")
    print(f"Total inference time: {results['total_inference_time']:.2f} seconds")
    timing_stats = results.get("timing_statistics", {}) or {}
    summary_times = timing_stats.get("summary_times", {}) or {}
    if "total_wall_time_s" in summary_times:
        summary = summary_times["total_wall_time_s"]
        print(
            "Profiled solve time: "
            f"{float(summary.get('total_s', 0.0) or 0.0):.2f} seconds total, "
            f"{float(summary.get('average_s', 0.0) or 0.0):.2f} seconds/sample"
        )
    module_times = timing_stats.get("module_times", {}) or {}
    if module_times:
        print("\nTiming Breakdown:")
        print("-" * 40)
        for name, stats in list(module_times.items())[:10]:
            total_s = float(stats.get("total_s", 0.0) or 0.0)
            avg_s = float(stats.get("average_s", 0.0) or 0.0)
            pct = float(stats.get("percent_of_observed_total", 0.0) or 0.0)
            print(f"{name:28s}: total {total_s:8.2f}s | avg {avg_s:7.2f}s | {pct:5.1f}%")
    tool_times = timing_stats.get("tool_times", {}) or {}
    if tool_times:
        print("\nTool Timing:")
        print("-" * 40)
        for name, stats in list(tool_times.items())[:10]:
            calls = int(stats.get("calls", 0) or 0)
            total_s = float(stats.get("total_tool_call_time_s", 0.0) or 0.0)
            avg_s = float(stats.get("average_tool_call_time_s", 0.0) or 0.0)
            print(f"{name:28s}: calls {calls:4d} | total {total_s:8.2f}s | avg {avg_s:7.2f}s")
    
    print("\nTask-wise Statistics:")
    print("-" * 40)
    for task, stats in results['task_statistics'].items():
        metric_name = stats.get("paper_metric", stats.get("metric", "ACC"))
        score = stats.get("paper_score", stats.get("score", stats.get("accuracy", 0.0)))
        if metric_name == "ACC":
            print(f"{task:20s}: {score:.4f} ({stats.get('correct', 0)}/{stats['total']}) [ACC]")
        else:
            print(
                f"{task:20s}: {score:.4f} "
                f"(perfect {stats.get('perfect', 0)}/{stats['total']}) [MRA]"
            )
            if "relaxed_score" in stats:
                print(f"{'':20s}  relaxed: {stats['relaxed_score']:.4f} [relaxed_MRA]")
    
    # Print correct and incorrect question IDs if available
    if 'correct_question_ids' in results and 'incorrect_question_ids' in results:
        print(f"\nPerfect-score questions: {len(results['correct_question_ids'])} IDs")
        print(f"Non-perfect questions: {len(results['incorrect_question_ids'])} IDs")
    
    if results['failed_samples_details']:
        print(f"\nFailed samples ({len(results['failed_samples_details'])}):")
        print("-" * 40)
        for failed in results['failed_samples_details'][:5]:  # 只显示前5个
            print(f"ID: {failed['id']}, Error: {failed['error']}")
        if len(results['failed_samples_details']) > 5:
            print(f"... and {len(results['failed_samples_details']) - 5} more")


def save_result_to_csv(result_data: Dict[str, Any], csv_file: str = "error_analysis.csv"):
    """保存结果信息到CSV文件

    Args:
        result_data: 包含结果信息的字典
        csv_file: CSV文件名
    """
    # 定义列名
    columns = ['question', 'path', 'is_correct', 'analysis', 'normalized_prediction', 'normalized_ground_truth', 'used_tools', 'skill_choices', 'follow_up_prompt']
    
    # 准备数据行
    row_data = {
        'question': result_data.get('question', ''),
        'path': result_data.get('path', ''),
        'is_correct': result_data.get('is_correct', ''),
        'analysis': result_data.get('analysis', ''),
        'normalized_prediction': result_data.get('normalized_prediction', ''),
        'normalized_ground_truth': result_data.get('normalized_ground_truth', ''),
        'used_tools': result_data.get('used_tools', ''),
        'skill_choices': result_data.get('skill_choices', ''),
        'follow_up_prompt': result_data.get('follow_up_prompt', '')
    }
    
    # 检查文件是否存在
    if os.path.exists(csv_file):
        # 如果文件存在，追加数据
        df_existing = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip', engine='python')
        df_new = pd.DataFrame([row_data])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
    else:
        # 如果文件不存在，创建新文件
        df_new = pd.DataFrame([row_data])
        df_new.to_csv(csv_file, index=False)

def extract_objects_from_response(response: str) -> list:
    """
    从回答中提取<object></object>标签包含的物体列表
    
    Args:
        response: VLLM的回答文本
        
    Returns:
        提取的物体列表
    """
    import logging
    import re
    
    logger = logging.getLogger(__name__)
    objects = []
    try:
        # 查找所有带编号的object标签对
        pattern = r'<object_\d+>(.*?)</object_\d+>'
        matches = re.findall(pattern, response)
        
        # 清理并添加到列表
        for match in matches:
            obj = match.strip()
            if obj:  # 只添加非空物体
                objects.append(obj)
                
        logger.info(f"从回答中提取到 {len(objects)} 个物体: {objects}")
    except Exception as e:
        logger.error(f"提取物体时出错: {e}")
    
    return objects

def draw_boxes_on_image(image_path: str, prompts: Dict, output_path: str = None) -> str:
    """
    在图像上绘制边界框
    
    Args:
        image_path: 输入图像路径
        prompts: 包含边界框坐标的字典，格式如 {'box': [[x1,y1,x2,y2], ...], 'labels': ['person', 'kite', ...]}
        output_path: 输出图像路径，如果为None则自动生成
        
    Returns:
        输出图像的路径
    """
    import cv2
    import numpy as np
    import os
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 如果没有指定输出路径，自动生成
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"outputs/boxes_{base_name}.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 绘制边界框
    if 'box' in prompts and prompts['box']:
        boxes = prompts['box']
        labels = prompts.get('labels', [])  # 获取标签列表，如果没有则为空列表
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            color = colors[i % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 添加标签（优先使用类别标签，否则使用默认标签）
            if i < len(labels) and labels[i]:
                label = labels[i]
            else:
                label = f"Box {i+1}"
            
            # 计算标签文本大小
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # 绘制标签背景
            cv2.rectangle(image, (x1, y1-label_size[1]-10), (x1+label_size[0]+10, y1), color, -1)
            
            # 绘制标签文本
            cv2.putText(image, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 保存图像
    cv2.imwrite(output_path, image)
    
    return output_path

def parse_json(json_output: str) -> str:
    """
    Parse JSON output by removing markdown fencing
    Based on qwen's official implementation
    
    Args:
        json_output: Raw response that may contain ```json fencing
        
    Returns:
        Clean JSON string
    """
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output
