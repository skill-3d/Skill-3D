#!/usr/bin/env python3
"""
SPAgent 快速开始演示脚本
Quick Start Demo for SPAgent

这个脚本展示如何快速上手使用新的SPAgent系统
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_1_basic_usage():
    """演示1: 基础使用"""
    print("\n" + "="*80)
    print("🚀 演示1: SPAgent基础使用")
    print("="*80)
    
    try:
        from skill3d import SPAgent
        from models import GPTModel
        from tools import DepthEstimationTool
        
        # 创建模型
        print("1. 创建GPT模型...")
        model = GPTModel(model_name="gpt-4o-mini")
        
        # 创建工具
        print("2. 创建深度估计工具...")
        tools = [DepthEstimationTool(use_mock=True)]
        
        # 创建智能体
        print("3. 创建SPAgent...")
        agent = SPAgent(model=model, tools=tools)
        
        # 查看工具
        print(f"4. 当前可用工具: {agent.list_tools()}")
        
        print("✅ 基础设置完成！")
        
        return agent
        
    except Exception as e:
        print(f"❌ 演示1失败: {e}")
        print("请确保已正确安装依赖和配置API密钥")
        return None

def demo_2_multi_tools():
    """演示2: 多工具组合"""
    print("\n" + "="*80)
    print("🔧 演示2: 多工具组合使用")
    print("="*80)
    
    try:
        from skill3d import SPAgent
        from models import GPTModel
        from tools import (
            DepthEstimationTool,
            SegmentationTool,
            ObjectDetectionTool
        )
        
        # 创建多工具智能体
        print("1. 创建包含多个工具的智能体...")
        model = GPTModel(model_name="gpt-4o-mini")
        tools = [
            DepthEstimationTool(use_mock=True),
            SegmentationTool(use_mock=True),
            ObjectDetectionTool(use_mock=True)
        ]
        
        agent = SPAgent(model=model, tools=tools)
        
        print(f"2. 智能体配置完成，包含 {len(tools)} 个工具:")
        for tool_name in agent.list_tools():
            print(f"   - {tool_name}")
        
        print("✅ 多工具智能体创建成功！")
        
        return agent
        
    except Exception as e:
        print(f"❌ 演示2失败: {e}")
        return None

def demo_3_dynamic_management():
    """演示3: 动态工具管理"""
    print("\n" + "="*80)
    print("⚡ 演示3: 动态工具管理")
    print("="*80)
    
    try:
        from skill3d import SPAgent
        from models import GPTModel
        from tools import DepthEstimationTool, SegmentationTool, ObjectDetectionTool
        
        # 从空智能体开始
        print("1. 创建空智能体...")
        agent = SPAgent(model=GPTModel())
        print(f"   初始工具: {agent.list_tools()}")
        
        # 动态添加工具
        print("2. 动态添加工具...")
        agent.add_tool(DepthEstimationTool(use_mock=True))
        print(f"   添加深度工具后: {agent.list_tools()}")
        
        agent.add_tool(SegmentationTool(use_mock=True))
        print(f"   添加分割工具后: {agent.list_tools()}")
        
        agent.add_tool(ObjectDetectionTool(use_mock=True))
        print(f"   添加检测工具后: {agent.list_tools()}")
        
        # 移除工具
        print("3. 移除工具...")
        agent.remove_tool("detect_objects_tool")
        print(f"   移除检测工具后: {agent.list_tools()}")
        
        print("✅ 动态工具管理演示完成！")
        
        return agent
        
    except Exception as e:
        print(f"❌ 演示3失败: {e}")
        return None

def demo_4_problem_solving():
    """演示4: 实际问题解决"""
    print("\n" + "="*80)
    print("🎯 演示4: 实际问题解决")
    print("="*80)
    
    try:
        from skill3d import SPAgent
        from models import GPTModel
        from tools import DepthEstimationTool, SegmentationTool
        
        # 创建智能体
        model = GPTModel(model_name="gpt-4o-mini")
        tools = [
            DepthEstimationTool(use_mock=True),
            SegmentationTool(use_mock=True)
        ]
        agent = SPAgent(model=model, tools=tools)
        
        # 模拟图像路径
        image_path = "assets/example.png"  # 这个是示例路径
        
        print("1. 准备解决问题...")
        print(f"   图像路径: {image_path}")
        print("   问题: 分析这张图片的深度关系和主要对象")
        
        # 注意：这里使用mock模式，所以不需要真实图像
        if not os.path.exists(image_path):
            print("   注意: 使用mock模式演示（无需真实图像）")
            # 创建一个虚拟图像文件用于演示
            os.makedirs("assets", exist_ok=True)
            with open(image_path, "w") as f:
                f.write("dummy image file for demo")
        
        print("2. 开始问题解决...")
        question = "分析这张图片的深度关系和主要对象"
        
        # 解决问题
        result = agent.solve_problem(image_path, question)
        
        print("3. 问题解决完成！")
        print(f"   工具调用数量: {len(result['tool_calls'])}")
        print(f"   使用的工具: {result['used_tools']}")
        print(f"   生成的额外图像: {len(result['additional_images'])}")
        print(f"   回答预览: {result['answer'][:100]}...")
        
        print("✅ 问题解决演示完成！")
        
        # 清理演示文件
        if os.path.exists(image_path) and os.path.getsize(image_path) < 100:
            os.remove(image_path)
        
        return result
        
    except Exception as e:
        print(f"❌ 演示4失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_5_tool_specialization():
    """演示5: 工具专门化配置"""
    print("\n" + "="*80)
    print("🎨 演示5: 不同场景的工具配置")
    print("="*80)
    
    try:
        from skill3d import SPAgent
        from models import GPTModel
        from tools import (
            DepthEstimationTool,
            SegmentationTool,
            ObjectDetectionTool,
            SupervisionTool
        )
        
        model = GPTModel(model_name="gpt-4o-mini")
        
        # 场景1: 深度分析专用
        print("1. 深度分析专用智能体:")
        depth_tools = [DepthEstimationTool(use_mock=True), SegmentationTool(use_mock=True)]
        depth_agent = SPAgent(model=model, tools=depth_tools)
        print(f"   工具配置: {depth_agent.list_tools()}")
        
        # 场景2: 目标检测专用
        print("2. 目标检测专用智能体:")
        detection_tools = [ObjectDetectionTool(use_mock=True), SupervisionTool(use_mock=True)]
        detection_agent = SPAgent(model=model, tools=detection_tools)
        print(f"   工具配置: {detection_agent.list_tools()}")
        
        # 场景3: 全功能智能体
        print("3. 全功能智能体:")
        all_tools = [
            DepthEstimationTool(use_mock=True),
            SegmentationTool(use_mock=True),
            ObjectDetectionTool(use_mock=True),
            SupervisionTool(use_mock=True)
        ]
        full_agent = SPAgent(model=model, tools=all_tools, max_workers=4)
        print(f"   工具配置: {full_agent.list_tools()}")
        print(f"   并行工作线程: 4")
        
        print("✅ 工具专门化配置演示完成！")
        
        return {"depth": depth_agent, "detection": detection_agent, "full": full_agent}
        
    except Exception as e:
        print(f"❌ 演示5失败: {e}")
        return None

def print_usage_summary():
    """打印使用总结"""
    print("\n" + "="*80)
    print("📚 SPAgent使用总结")
    print("="*80)
    
    print("""
🎯 核心概念:
1. SPAgent = Model + Tools + 问题解决逻辑
2. Model: GPTModel, QwenModel, QwenVLLMModel
3. Tools: 各种专家工具的封装

🚀 基本使用流程:
1. 创建模型: model = GPTModel()
2. 创建工具: tools = [DepthEstimationTool(), ...]
3. 创建智能体: agent = SPAgent(model, tools)
4. 解决问题: result = agent.solve_problem(image_path, question)

🔧 工具管理:
- 添加工具: agent.add_tool(tool)
- 移除工具: agent.remove_tool(tool_name)
- 查看工具: agent.list_tools()
- 更换模型: agent.set_model(new_model)

💡 使用技巧:
- 使用 use_mock=True 进行快速测试
- 根据具体任务选择合适的工具组合
- 利用并行执行提高性能 (max_workers参数)
- 支持单图或多图分析

⚡ 性能优势:
- 自动并行工具执行
- 模块化设计，易于扩展
- 比旧workflow系统减少90%代码
""")

def main():
    """主函数"""
    print("🎉 欢迎使用SPAgent快速开始演示！")
    print("本演示将展示SPAgent的核心功能和使用方法")
    
    # 检查基础依赖
    try:
        import openai
        print("✅ OpenAI库已安装")
    except ImportError:
        print("⚠️  OpenAI库未安装，某些功能可能不可用")
    
    # 运行演示
    demos = [
        ("基础使用", demo_1_basic_usage),
        ("多工具组合", demo_2_multi_tools),
        ("动态工具管理", demo_3_dynamic_management),
        ("实际问题解决", demo_4_problem_solving),
        ("工具专门化配置", demo_5_tool_specialization)
    ]
    
    results = {}
    for name, demo_func in demos:
        try:
            result = demo_func()
            results[name] = result
            if result is not None:
                print(f"✅ {name} 演示成功")
            else:
                print(f"⚠️  {name} 演示部分成功")
        except Exception as e:
            print(f"❌ {name} 演示失败: {e}")
            results[name] = None
    
    # 打印总结
    print_usage_summary()
    
    # 成功统计
    successful = sum(1 for r in results.values() if r is not None)
    total = len(results)
    
    print(f"\n🎯 演示完成: {successful}/{total} 个演示成功")
    
    if successful == total:
        print("🎉 所有演示都成功运行！你已经掌握了SPAgent的基本使用方法")
    elif successful > 0:
        print("👍 部分演示成功，请检查失败的演示以确保环境配置正确")
    else:
        print("❌ 所有演示都失败，请检查:")
        print("   1. Python环境和依赖安装")
        print("   2. API密钥配置")
        print("   3. 项目路径设置")
    
    print("\n📖 更多信息请查看:")
    print("   - README.md: 完整使用说明")
    print("   - MIGRATION_GUIDE.md: 迁移指南")
    print("   - examples/skill3d_example.py: 详细示例")

if __name__ == "__main__":
    main() 
