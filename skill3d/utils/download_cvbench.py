#!/usr/bin/env python3
"""
下载CV-Bench数据集并转换为JSONL格式
"""

import argparse
import json
import os
import re

def download_cvbench(test_mode=False, max_samples=5):
    """下载CV-Bench数据集并转换为JSONL格式"""
    from datasets import load_dataset
    
    print(f"开始下载CV-Bench数据集... {'(测试模式，只处理前' + str(max_samples) + '条数据)' if test_mode else ''}")
    
    # 加载CV-Bench数据集
    try:
        ds = load_dataset("nyu-visionx/CV-Bench", "default")
        print(f"数据集加载成功！包含分割: {list(ds.keys())}")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    
    # 创建dataset文件夹
    dataset_folder = "dataset"
    print(f"检查dataset文件夹: {os.path.abspath(dataset_folder)}")
    os.makedirs(dataset_folder, exist_ok=True)
    print(f"dataset文件夹存在: {os.path.exists(dataset_folder)}")
    
    # 处理所有分割的数据
    all_converted_data = []
    total_processed = 0
    
    # 遍历所有数据分割
    for split_name, split_data in ds.items():
        print(f"\n开始处理分割: {split_name}")
        print(f"  数据条目数: {len(split_data)}")
        
        # 在测试模式下限制处理的数据量
        if test_mode:
            # 为了测试2D和3D数据，我们取一些2D和3D的样本
            # 2D: idx 0-9, 3D: idx 1438-1447
            process_indices = list(range(0, 5)) + list(range(1438, 1443))
        else:
            process_indices = list(range(len(split_data)))
            
        process_count = len(process_indices)
        print(f"  将处理: {process_count} 条数据")
        
        for idx in process_indices:
            sample = split_data[idx]
            try:
                # 根据数据集的idx和type字段来确定对应的本地图片路径
                # 图片已经按照cvbench_img.py的逻辑保存在dataset/CVBench/2D和dataset/CVBench/3D中
                # 映射关系:
                # - 2D数据: HuggingFace idx 0-1437 → 本地文件 000000.png - 001437.png  
                # - 3D数据: HuggingFace idx 1438-2637 → 本地文件 000000.png - 001199.png
                
                # 获取样本的idx和type
                sample_idx = sample.get("idx", idx)
                sample_type = sample.get("type", "2D")  # 2D或3D
                
                # 计算本地文件的编号
                if sample_type == "2D":
                    # 2D数据直接使用idx
                    local_file_id = sample_idx
                elif sample_type == "3D":
                    # 3D数据需要减去1438
                    local_file_id = sample_idx - 1438
                else:
                    print(f"  警告: 未知的type: {sample_type}")
                    continue
                
                # 构建图片路径
                # 本地文件路径（用于验证文件存在）
                local_image_folder = f"dataset/CVBench/{sample_type}"
                image_filename = f"{local_file_id:06d}.png"
                local_image_path = f"{local_image_folder}/{image_filename}"
                
                # JSONL中保存的相对路径（不包含dataset/前缀）
                jsonl_image_path = f"CVBench/{sample_type}/{image_filename}"
                
                # 验证图片文件是否存在
                if not os.path.exists(local_image_path):
                    print(f"  警告: 图片文件不存在: {local_image_path} (HuggingFace idx: {sample_idx}, type: {sample_type})")
                    continue
                
                # 处理图片路径（JSONL中保存相对路径）
                image_paths = [jsonl_image_path]                # 构建对话内容
                conversations = []
                
                # 添加人类问题
                question_text = ""
                if "question" in sample and sample["question"]:
                    question_text = sample["question"]
                    
                    # 如果有选项，添加选项信息
                    if "choices" in sample and sample["choices"]:
                        question_text += "\nSelect from the following choices:\n"
                        for i, choice in enumerate(sample["choices"]):
                            question_text += f"({chr(65+i)}) {choice}\n"
                
                if question_text:
                    conversations.append({
                        "from": "human",
                        "value": question_text
                    })
                
                # 添加GPT回答
                if "answer" in sample and sample["answer"] is not None:
                    answer = sample["answer"]
                    
                    # 如果答案是数字索引，转换为字母
                    if isinstance(answer, int) and "choices" in sample and sample["choices"]:
                        if 0 <= answer < len(sample["choices"]):
                            answer = chr(65 + answer)  # 转换为A, B, C...
                    
                    # 如果答案是 "(A)" 格式，提取括号中的字母
                    elif isinstance(answer, str) and answer.startswith("(") and answer.endswith(")"):
                        answer = answer[1:-1]  # 去掉括号，只保留字母
                    
                    # 如果答案是choice中的文本，尝试转换为字母
                    elif "choices" in sample and sample["choices"] and answer in sample["choices"]:
                        answer_idx = sample["choices"].index(answer)
                        answer = chr(65 + answer_idx)  # 转换为A, B, C...
                    
                    conversations.append({
                        "from": "gpt", 
                        "value": str(answer)
                    })
                
                # 确定任务类型
                task_name = sample.get("task", sample.get("type", "unknown"))
                
                # 确定输入输出类型
                input_type = "image"
                output_type = "text"
                if "choices" in sample and sample["choices"]:
                    output_type = "MCQ"
                
                # 构建JSON条目
                json_entry = {
                    "id": f"{task_name}_{sample.get('idx', idx)}",
                    "image": image_paths,  # 图片路径列表
                    "video": [],  # video没有就空着
                    "conversations": conversations,
                    "task": task_name,
                    "input_type": input_type,
                    "output_type": output_type,
                    "data_source": "CVBench",
                    "others": {},
                    "subtask": ""  # subtask没有就放空
                }
                                
                all_converted_data.append(json_entry)
                total_processed += 1
                
                # 在测试模式下，如果达到最大样本数就停止
                if test_mode and total_processed >= max_samples:
                    print(f"    测试模式：已处理 {total_processed} 条数据，停止处理")
                    break
                
                if total_processed % 100 == 0:
                    print(f"    已处理 {total_processed} 条数据...")
                
            except Exception as e:
                print(f"  处理数据 {idx} 时出错: {e}")
                continue
        
        # 在测试模式下，如果达到最大样本数就停止处理其他分割
        if test_mode and total_processed >= max_samples:
            break
    
    print(f"\n所有数据处理完成！总共处理了 {total_processed} 条数据")
    
    # 保存JSONL文件（每个ID一行）
    json_filename = 'cvbench_test.jsonl' if test_mode else 'cvbench_data.jsonl'
    json_path = f'dataset/{json_filename}'
    print(f"准备保存JSONL文件到: {os.path.abspath(json_path)}")
    print(f"数据条目数量: {len(all_converted_data)}")
    
    if all_converted_data:
        with open(json_path, 'w', encoding='utf-8') as f:
            for item in all_converted_data:
                json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')
        print(f"JSONL文件保存成功!")
    else:
        print("警告: 没有数据需要保存!")
    
    print(f"\n数据转换完成！")
    print(f"总共处理了 {len(all_converted_data)} 条数据")
    print(f"JSONL文件保存为: {json_path}")
    print(f"图片路径引用: dataset/CVBench/2D/ 和 dataset/CVBench/3D/ 文件夹中的高质量PNG图片")
    
    # 统计每个任务的数据量
    task_counts = {}
    for item in all_converted_data:
        task = item["task"]
        task_counts[task] = task_counts.get(task, 0) + 1
    
    print("\n各任务数据统计:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} 条")
    
    # 统计输出类型
    output_type_counts = {}
    for item in all_converted_data:
        output_type = item["output_type"]
        output_type_counts[output_type] = output_type_counts.get(output_type, 0) + 1
    
    print("\n输出类型统计:")
    for output_type, count in sorted(output_type_counts.items()):
        print(f"  {output_type}: {count} 条")
    
    # 打印第一条数据作为示例
    if all_converted_data:
        print("\n第一条数据示例:")
        print(json.dumps(all_converted_data[0], ensure_ascii=False, indent=2))


def _read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def _normalize_cvbench_answer(answer, choices):
    if answer is None:
        return ""

    if isinstance(answer, int) and choices:
        if 0 <= answer < len(choices):
            return chr(65 + answer)

    answer_text = str(answer).strip()
    letter_match = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?", answer_text)
    if letter_match:
        return letter_match.group(1).upper()

    letter_match = re.search(r"\(([A-Za-z])\)", answer_text)
    if letter_match:
        return letter_match.group(1).upper()

    if choices and answer_text in choices:
        return chr(65 + choices.index(answer_text))

    return answer_text


def _build_cvbench_question(sample):
    prompt = sample.get("prompt")
    if prompt:
        return str(prompt).strip()

    question = str(sample.get("question", "")).strip()
    choices = sample.get("choices") or []
    if choices:
        question += "\nSelect from the following choices.\n"
        question += "\n".join(f"({chr(65 + idx)}) {choice}" for idx, choice in enumerate(choices))
    return question


def convert_local_cvbench(
    raw_2d_path="dataset/test_2d.jsonl",
    raw_3d_path="dataset/test_3d.jsonl",
    image_root="dataset/CVBench",
    output_path="dataset/cvbench_data.jsonl",
    test_mode=False,
    max_samples=10,
    strict=False,
):
    """Convert locally saved CV-Bench JSONL files into SPAgent format.

    The local images are expected to follow the naming produced by
    ``cvbench_img.py``:

    - dataset/CVBench/2D/000000.png ... 001437.png
    - dataset/CVBench/3D/000000.png ... 001199.png
    """
    raw_specs = [("2D", raw_2d_path), ("3D", raw_3d_path)]
    output_path = "dataset/cvbench_test.jsonl" if test_mode and output_path == "dataset/cvbench_data.jsonl" else output_path
    all_converted_data = []
    missing_images = []

    print("开始从本地JSONL转换CV-Bench数据集...")
    for sample_type, raw_path in raw_specs:
        if not os.path.exists(raw_path):
            message = f"原始文件不存在: {raw_path}"
            if strict:
                raise FileNotFoundError(message)
            print(f"  警告: {message}")
            continue

        records = _read_jsonl(raw_path)
        print(f"  读取 {sample_type}: {raw_path}, {len(records)} 条")

        for local_idx, sample in enumerate(records):
            if test_mode and len(all_converted_data) >= max_samples:
                break

            record_type = str(sample.get("type") or sample_type)
            if record_type not in {"2D", "3D"}:
                record_type = sample_type

            image_filename = f"{local_idx:06d}.png"
            local_image_path = os.path.join(image_root, record_type, image_filename)
            jsonl_image_path = f"CVBench/{record_type}/{image_filename}"
            if not os.path.exists(local_image_path):
                missing_images.append(local_image_path)
                if strict:
                    raise FileNotFoundError(f"图片文件不存在: {local_image_path}")
                continue

            choices = sample.get("choices") or []
            question_text = _build_cvbench_question(sample)
            answer = _normalize_cvbench_answer(sample.get("answer"), choices)
            task_name = str(sample.get("task") or record_type)

            json_entry = {
                "id": f"CVBench_{record_type}_{local_idx:06d}",
                "image": [jsonl_image_path],
                "video": [],
                "conversations": [
                    {"from": "human", "value": question_text},
                    {"from": "gpt", "value": answer},
                ],
                "task": task_name,
                "input_type": "image",
                "output_type": "MCQ" if choices else "text",
                "data_source": "CVBench",
                "sub_task": "",
                "others": {
                    "type": record_type,
                    "local_index": local_idx,
                    "original_answer": sample.get("answer"),
                    "choices": choices,
                    "original_filename": sample.get("filename"),
                    "source": sample.get("source"),
                    "source_dataset": sample.get("source_dataset"),
                    "source_filename": sample.get("source_filename"),
                    "target_class": sample.get("target_class"),
                    "target_size": sample.get("target_size"),
                    "bbox": sample.get("bbox"),
                },
            }
            all_converted_data.append(json_entry)

        if test_mode and len(all_converted_data) >= max_samples:
            break

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_converted_data:
            json.dump(item, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")

    print(f"\n转换完成: {output_path}")
    print(f"总条数: {len(all_converted_data)}")
    if missing_images:
        print(f"跳过缺失图片样本: {len(missing_images)}")
        print(f"第一条缺失图片: {missing_images[0]}")

    task_counts = {}
    type_counts = {}
    max_choice_count = 0
    for item in all_converted_data:
        task_counts[item["task"]] = task_counts.get(item["task"], 0) + 1
        item_type = item["others"]["type"]
        type_counts[item_type] = type_counts.get(item_type, 0) + 1
        max_choice_count = max(max_choice_count, len(item["others"].get("choices") or []))

    print("\n类型统计:")
    for item_type, count in sorted(type_counts.items()):
        print(f"  {item_type}: {count} 条")

    print("\n任务统计:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} 条")

    print(f"\n最大选项数量: {max_choice_count}")
    if all_converted_data:
        print("\n第一条数据示例:")
        print(json.dumps(all_converted_data[0], ensure_ascii=False, indent=2))

    return all_converted_data


def main():
    parser = argparse.ArgumentParser(description="Convert CV-Bench into SPAgent JSONL format.")
    parser.add_argument("--source", choices=["auto", "local", "hf"], default="auto",
                        help="auto优先使用本地test_2d/test_3d JSONL，缺失时再走HuggingFace下载")
    parser.add_argument("--raw_2d_path", default="dataset/test_2d.jsonl")
    parser.add_argument("--raw_3d_path", default="dataset/test_3d.jsonl")
    parser.add_argument("--image_root", default="dataset/CVBench")
    parser.add_argument("--output_path", default="dataset/cvbench_data.jsonl")
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    use_local = args.source == "local" or (
        args.source == "auto"
        and os.path.exists(args.raw_2d_path)
        and os.path.exists(args.raw_3d_path)
    )

    if use_local:
        convert_local_cvbench(
            raw_2d_path=args.raw_2d_path,
            raw_3d_path=args.raw_3d_path,
            image_root=args.image_root,
            output_path=args.output_path,
            test_mode=args.test_mode,
            max_samples=args.max_samples,
            strict=args.strict,
        )
    else:
        download_cvbench(test_mode=args.test_mode, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
