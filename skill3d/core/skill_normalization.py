"""
Skill normalization helpers.
"""

from typing import Dict, Any, List

from .skill import Skill, SkillRegistry


def build_default_skills() -> SkillRegistry:
    registry = SkillRegistry()

    registry.register(Skill(
        name="Skill 1: Spatial Localization & Relation",
        when='where / relative position / between / left-right / spatial relationship',
        strategy=[
            "Identify the relevant objects and relation first.",
            "Use any localization-capable tool that can best reduce uncertainty for this scene.",
            "If orientation or facing direction is the core uncertainty, prioritize orient_anything_tool on the target crop instead of treating it as a late verification step.",
            "Cross-check with orientation, segmentation, or 3D evidence only when it may change the answer.",
        ],
        tool_usage="Common tools: orient_anything_tool, detect_objects_tool, moondream_tool, segment_image_tool, pi3_tool",
    ))

    registry.register(Skill(
        name="Skill 2: Depth & Distance Reasoning",
        when="closer/farther / distance / front-back / which is nearer",
        strategy=[
            "When the question is about nearest/farthest, front/back, closest-point distance, or depth ranking, treat depth_estimation_tool as the anchor first-pass tool rather than a weak auxiliary cue.",
            "Do not default to detect_objects_tool on every sample. If reusable crop boxes are already clear, go straight to depth_estimation_tool with crop_box, crop_boxes, point_coords, or point_pairs.",
            "If the unresolved blocker is exact target points, closest points, or object boundaries, use moondream_tool or segment_image_tool before assuming a coarse box is sufficient.",
            "If the relevant object region is tiny or blurry, enhance the local crop with swinir_tool before repeating localization or depth.",
            "Use detect_objects_tool only when you genuinely need reusable category-level boxes for depth_estimation_tool, swinir_tool, or orient_anything_tool.",
            "Only escalate to heavier geometry tools such as pi3_tool after direct 2D cues remain ambiguous.",
        ],
        tool_usage="Common tools: depth_estimation_tool, segment_image_tool, moondream_tool, swinir_tool, detect_objects_tool, pi3_tool",
    ))

    registry.register(Skill(
        name="Skill 3: Multi-view Verification & Occlusion",
        when="camera viewpoints / occlusion / visibility / which camera can see",
        strategy=[
            "Use viewpoint-changing tools when visibility or occlusion is the core uncertainty.",
            "Compare multiple rendered views or camera references when that helps resolve what is visible.",
            "Run detection or pointing on rendered views if the answer depends on a specific object.",
        ],
        tool_usage="Common tools: pi3_tool, detect_objects_tool, moondream_tool, orient_anything_tool",
    ))

    registry.register(Skill(
        name="Skill 4: Object Detection & Counting",
        when="how many / count / identify all objects",
        strategy=[
            "Start from the tool that can most directly reveal the target instances.",
            "If a target is tiny, blurry, far away, or low-resolution, treat swinir_tool as the preferred helper before deciding from raw pixels.",
            "Use zoom enhancement, open-vocabulary detection, or custom-category detection when helpful.",
            "Verify difficult cases with pointing or segmentation if overlap or ambiguity remains.",
        ],
        tool_usage="Common tools: supervision_tool, yoloe_detection_tool, detect_objects_tool, swinir_tool, moondream_tool, segment_image_tool",
    ))

    registry.register(Skill(
        name="Skill 5: Object Segmentation & Boundary Analysis",
        when="segment / boundary / shape / outline",
        strategy=[
            "Localize the target region first if needed.",
            "Use segmentation when mask quality or exact boundary matters.",
            "Visualize or cross-check the mask if the boundary decision is important.",
        ],
        tool_usage="Common tools: detect_objects_tool, yoloe_detection_tool, segment_image_tool, supervision_tool",
    ))

    registry.register(Skill(
        name="Skill 6: Fine-grained Object Localization",
        when="point to / locate exactly / multi-object pointing",
        strategy=[
            "Use the tool that can most precisely localize the target object.",
            "Enhance local detail if the target is small or blurry.",
            "A useful pattern is detect_objects_tool first, then apply swinir_tool on the detected crop, and only then make the fine localization decision.",
            "Use boxes or masks only if they improve the final precision.",
        ],
        tool_usage="Common tools: moondream_tool, detect_objects_tool, swinir_tool, segment_image_tool",
    ))

    registry.register(Skill(
        name="Skill 7: Custom Category Detection",
        when="detect all X where X is specific",
        strategy=[
            "Use custom-category detection when the target class is explicitly specified.",
            "Verify borderline detections with another tool if confidence is low.",
            "Visualize the result when it helps confirm coverage.",
        ],
        tool_usage="Common tools: yoloe_detection_tool, moondream_tool, supervision_tool, swinir_tool",
    ))

    registry.register(Skill(
        name="Skill 8: Comprehensive Scene Understanding",
        when="complex questions requiring multiple signals",
        strategy=[
            "Break the question into the smallest evidence needs.",
            "Choose the tool mix that best covers overview, object evidence, geometry, and verification.",
            "If one evidence need is nearest/farthest, front/back, closest-point distance, or depth ranking, keep depth_estimation_tool as a strong anchor candidate once the target objects or points are clear.",
            "Do not let detect_objects_tool become the automatic opener. Pick it only when reusable boxes are the actual missing evidence.",
            "When the unresolved blocker is boundary precision, exact points, or local detail, prefer segment_image_tool, moondream_tool, or swinir_tool before escalating to heavier tools. Treat swinir_tool as the default recovery path for unreadable local crops.",
            "If orientation or pose is the missing signal, prefer orient_anything_tool before escalating to 3D. Delay pi3_tool until direct 2D evidence still leaves viewpoint or occlusion ambiguity.",
            "Add complementary tools only when they can change the answer.",
        ],
        tool_usage="Common tools: supervision_tool, depth_estimation_tool, segment_image_tool, moondream_tool, swinir_tool, detect_objects_tool, yoloe_detection_tool, orient_anything_tool, pi3_tool",
    ))

    return registry


def skills_to_json(skills: SkillRegistry) -> List[Dict[str, Any]]:
    return skills.to_json_list()
