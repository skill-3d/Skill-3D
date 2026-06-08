import sys
import types
import unittest
import importlib.machinery
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
cv2_stub = sys.modules.setdefault("cv2", types.ModuleType("cv2"))
if getattr(cv2_stub, "__spec__", None) is None:
    cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)

from plugin.plugin_all_angles import SPAgentToolCallingScheduler


class DummyInferRequest:
    def __init__(self, images):
        self.images = images
        self.data_dict = {"original_images": images}


class TestPluginAllAnglesToolCallRepair(unittest.TestCase):
    def test_parse_nested_tool_payload(self):
        scheduler = SPAgentToolCallingScheduler(max_turns=3)
        content = (
            '<tool_call>'
            '{"tool":{"name":"segment_image_tool","params":{"image_index":3,"objects_to_segment":["wall"],"confidence_threshold":0.7}}}'
            '</tool_call>'
        )
        calls = scheduler._parse_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "segment_image_tool")
        self.assertIn("image_index", calls[0]["arguments"])

    def test_fix_segment_image_index_to_image_path(self):
        scheduler = SPAgentToolCallingScheduler(max_turns=3)
        infer_request = DummyInferRequest(["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"])
        fixed = scheduler._validate_and_fix_tool_calls(
            [
                {
                    "name": "segment_image_tool",
                    "arguments": {
                        "image_index": 3,
                        "objects_to_segment": ["wall"],
                        "confidence_threshold": 0.7,
                    },
                }
            ],
            infer_request,
        )
        self.assertEqual(len(fixed), 1)
        self.assertEqual(fixed[0]["arguments"]["image_path"], "img2.jpg")
        self.assertNotIn("image_index", fixed[0]["arguments"])
        self.assertNotIn("confidence_threshold", fixed[0]["arguments"])

    def test_continuation_prompt_ignores_none_additional_images(self):
        scheduler = SPAgentToolCallingScheduler(max_turns=3)
        prompt = scheduler._create_continuation_prompt(
            question="Where is the chair?",
            last_response="<think>x</think><tool_call>{}</tool_call>",
            tool_results={"detect_objects_tool": {"success": True}},
            original_images=["img0.jpg", None, "img1.jpg"],
            additional_images=[None, "outputs/vis.png"],
            current_turn=1,
            max_turns=3,
        )
        self.assertIn("vis.png", prompt)
        self.assertNotIn("NoneType", prompt)

    def test_fix_detect_objects_prompt_aliases(self):
        scheduler = SPAgentToolCallingScheduler(max_turns=3)
        infer_request = DummyInferRequest(["img0.jpg"])
        fixed = scheduler._validate_and_fix_tool_calls(
            [
                {
                    "name": "detect_objects_tool",
                    "arguments": {
                        "image_path": "img0.jpg",
                        "prompt": "chair, table",
                    },
                }
            ],
            infer_request,
        )
        self.assertEqual(fixed[0]["arguments"]["text_prompt"], "chair, table")
        self.assertNotIn("prompt", fixed[0]["arguments"])


if __name__ == "__main__":
    unittest.main()
