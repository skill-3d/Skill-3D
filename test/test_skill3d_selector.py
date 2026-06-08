import sys
import tempfile
import unittest
from pathlib import Path
import types


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from skill3d.core.model import Model
from skill3d.core.agent import SPAgent
from skill3d.core.skill_learning import AdaptiveSkillManager
from skill3d.core.tool import Tool


class DummyModel(Model):
    def single_image_inference(self, image_path, prompt, temperature=None, max_tokens=None) -> str:
        return ""

    def multiple_images_inference(self, image_paths, prompt, temperature=None, max_tokens=None) -> str:
        return ""

    def text_only_inference(self, prompt, temperature=None, max_tokens=None) -> str:
        return ""


class DummyTool(Tool):
    def __init__(self, name: str, parameters):
        super().__init__(name=name, description="dummy")
        self._parameters = parameters

    @property
    def parameters(self):
        return self._parameters

    def call(self, **kwargs):
        return {"success": True, "result": kwargs}


class TestSkill3DSelector(unittest.TestCase):
    def _build_agent_and_bundle(self, question: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = AdaptiveSkillManager(
                storage_path=str(root / "learned_skills.json"),
                question_memory_export_path=str(root / "questions.json"),
            )
            tools = [
                DummyTool(
                    "detect_objects_tool",
                    {
                        "type": "object",
                        "properties": {
                            "image_path": {"type": "string"},
                            "text_prompt": {"type": "string"},
                        },
                        "required": ["image_path", "text_prompt"],
                    },
                ),
                DummyTool(
                    "depth_estimation_tool",
                    {
                        "type": "object",
                        "properties": {
                            "image_path": {"type": "string"},
                        },
                        "required": ["image_path"],
                    },
                ),
                DummyTool(
                    "pi3_tool",
                    {
                        "type": "object",
                        "properties": {
                            "image_path": {"type": "array"},
                            "azimuth_angle": {"type": "number"},
                            "elevation_angle": {"type": "number"},
                            "rotation_reference_camera": {"type": "integer"},
                            "camera_view": {"type": "boolean"},
                        },
                        "required": ["image_path"],
                    },
                ),
                DummyTool(
                    "moondream_tool",
                    {
                        "type": "object",
                        "properties": {
                            "image_path": {"type": "string"},
                            "task": {"type": "string"},
                            "object_name": {"type": "string"},
                        },
                        "required": ["image_path", "task", "object_name"],
                    },
                ),
            ]
            agent = SPAgent(
                model=DummyModel("dummy"),
                tools=tools,
                prompt_style="skills",
                skill_manager=manager,
            )
            bundle = manager.build_prompt_bundle(question)
            return agent, bundle

    def test_selector_blocks_detect_when_depth_must_try_missing(self):
        question = "Measuring from the closest point of each object, what is the distance between the chair and the sofa (in meters)?"
        agent, bundle = self._build_agent_and_bundle(question)
        approved, rejected, decision, note = agent._select_tool_calls(
            tool_calls=[
                {
                    "name": "detect_objects_tool",
                    "arguments": {"image_path": "frame.jpg", "text_prompt": "chair, sofa"},
                }
            ],
            skill_choices=[{"id": "seed::depth_distance", "source": "static"}],
            skill_bundle=bundle,
            question=question,
            all_tool_calls=[],
            all_tool_results={},
            iteration=1,
            view_state={"tried_views": [], "tried_reference_cameras": [], "view_gain_history": []},
        )

        self.assertEqual(approved, [])
        self.assertEqual(len(rejected), 1)
        self.assertIn("depth_estimation_tool", note)
        self.assertEqual(decision["missing_must_try_tools"], ["depth_estimation_tool"])

    def test_selected_detection_skill_requires_detection_tool_before_answer(self):
        question = "How many chair(s) are in this room?"
        agent, bundle = self._build_agent_and_bundle(question)

        missing = agent._missing_must_try_tools(
            bundle,
            tool_calls=[],
            skill_choices=[{"id": "seed::detection_counting", "source": "static"}],
        )

        self.assertEqual(missing, ["detect_objects_tool"])

        approved, rejected, decision, note = agent._select_tool_calls(
            tool_calls=[
                {
                    "name": "moondream_tool",
                    "arguments": {"image_path": "frame.jpg", "object_name": "chair"},
                }
            ],
            skill_choices=[{"id": "seed::detection_counting", "source": "static"}],
            skill_bundle=bundle,
            question=question,
            all_tool_calls=[],
            all_tool_results={},
            iteration=1,
            view_state={"tried_views": [], "tried_reference_cameras": [], "view_gain_history": []},
        )

        self.assertEqual(rejected, [])
        self.assertEqual(approved[0]["name"], "moondream_tool")
        self.assertEqual(decision["skill_required_tools"], ["detect_objects_tool"])
        self.assertEqual(decision["missing_must_try_tools"], ["detect_objects_tool"])
        self.assertIn("detect_objects_tool", note)

    def test_none_skill_does_not_require_skill_bound_tools(self):
        question = "How many chair(s) are in this room?"
        agent, bundle = self._build_agent_and_bundle(question)

        missing = agent._missing_must_try_tools(
            bundle,
            tool_calls=[],
            skill_choices=[{"id": "none", "source": "none"}],
        )

        self.assertEqual(missing, [])

    def test_selector_rejects_pi3_origin_view(self):
        question = "Would the vase become visible from another view behind the shelf?"
        agent, bundle = self._build_agent_and_bundle(question)
        approved, rejected, _, _ = agent._select_tool_calls(
            tool_calls=[
                {
                    "name": "pi3_tool",
                    "arguments": {"image_path": ["frame.jpg"], "azimuth_angle": 0, "elevation_angle": 0},
                }
            ],
            skill_choices=[{"id": "seed::multiview_occlusion", "source": "static"}],
            skill_bundle=bundle,
            question=question,
            all_tool_calls=[],
            all_tool_results={},
            iteration=1,
            view_state={"tried_views": [], "tried_reference_cameras": [], "view_gain_history": []},
        )

        self.assertEqual(approved, [])
        self.assertEqual(rejected[0]["failure_type"], "repeated_view")

    def test_selector_repairs_moondream_task_from_object_name(self):
        question = "Point to the mug."
        agent, bundle = self._build_agent_and_bundle(question)
        approved, rejected, _, _ = agent._select_tool_calls(
            tool_calls=[
                {
                    "name": "moondream_tool",
                    "arguments": {"image_path": "frame.jpg", "object_name": "mug"},
                }
            ],
            skill_choices=[{"id": "seed::fine_grained_pointing", "source": "static"}],
            skill_bundle=bundle,
            question=question,
            all_tool_calls=[],
            all_tool_results={},
            iteration=1,
            view_state={"tried_views": [], "tried_reference_cameras": [], "view_gain_history": []},
        )

        self.assertEqual(rejected, [])
        self.assertEqual(approved[0]["arguments"]["task"], "point")

    def test_selector_rejects_orient_anything_without_crop_box(self):
        question = "Which way is the chair facing?"
        agent, bundle = self._build_agent_and_bundle(question)
        orient_tool = DummyTool(
            "orient_anything_tool",
            {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "crop_box": {"type": "array"},
                },
                "required": ["image_path", "crop_box"],
            },
        )
        agent.remove_tool("orient_anything_tool")
        agent.add_tool(orient_tool)

        approved, rejected, _, _ = agent._select_tool_calls(
            tool_calls=[
                {
                    "name": "orient_anything_tool",
                    "arguments": {"image_path": "frame.jpg"},
                }
            ],
            skill_choices=[{"id": "seed::spatial_localization", "source": "static"}],
            skill_bundle=bundle,
            question=question,
            all_tool_calls=[],
            all_tool_results={},
            iteration=1,
            view_state={"tried_views": [], "tried_reference_cameras": [], "view_gain_history": []},
        )

        self.assertEqual(approved, [])
        self.assertEqual(rejected[0]["failure_type"], "schema_error")
        self.assertIn("crop_box", rejected[0]["selector_reason"])

    def test_selector_rejects_swinir_without_crop_for_localized_problem(self):
        question = "Which tiny blurry object is closer to the camera, the chair or the sofa?"
        agent, bundle = self._build_agent_and_bundle(question)
        swinir_tool = DummyTool(
            "swinir_tool",
            {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "crop_box": {"type": "array"},
                    "task": {"type": "string"},
                },
                "required": ["image_path"],
            },
        )
        agent.remove_tool("swinir_tool")
        agent.add_tool(swinir_tool)

        approved, rejected, _, _ = agent._select_tool_calls(
            tool_calls=[
                {
                    "name": "swinir_tool",
                    "arguments": {"image_path": "frame.jpg", "task": "real_sr"},
                }
            ],
            skill_choices=[{"id": "seed::depth_distance", "source": "static"}],
            skill_bundle=bundle,
            question=question,
            all_tool_calls=[],
            all_tool_results={},
            iteration=1,
            view_state={"tried_views": [], "tried_reference_cameras": [], "view_gain_history": []},
        )

        self.assertEqual(approved, [])
        self.assertEqual(rejected[0]["failure_type"], "selector_rejected")
        self.assertIn("crop_box", rejected[0]["selector_reason"])

    def test_selector_rejects_depth_only_image_for_closest_point_metric_question(self):
        question = "Measuring from the closest point of each object, what is the distance between the chair and the sofa (in meters)?"
        agent, bundle = self._build_agent_and_bundle(question)
        approved, rejected, _, _ = agent._select_tool_calls(
            tool_calls=[
                {
                    "name": "depth_estimation_tool",
                    "arguments": {"image_path": "frame.jpg"},
                }
            ],
            skill_choices=[{"id": "seed::depth_distance", "source": "static"}],
            skill_bundle=bundle,
            question=question,
            all_tool_calls=[],
            all_tool_results={},
            iteration=1,
            view_state={"tried_views": [], "tried_reference_cameras": [], "view_gain_history": []},
        )

        self.assertEqual(approved, [])
        self.assertEqual(rejected[0]["failure_type"], "schema_error")
        self.assertIn("point_coords", rejected[0]["selector_reason"])


if __name__ == "__main__":
    unittest.main()
