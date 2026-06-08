"""
Skill definitions and registry.
"""

from typing import Dict, Any, List, Optional


class Skill:
    def __init__(self, name: str, when: str, strategy: List[str], tool_usage: str):
        self.name = name
        self.when = when
        self.strategy = strategy
        self.tool_usage = tool_usage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "when": self.when,
            "strategy": self.strategy,
            "tool_usage": self.tool_usage,
        }


class SkillRegistry:
    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill):
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def list_skills(self) -> List[str]:
        return list(self._skills.keys())

    def get_all_skills(self) -> Dict[str, Skill]:
        return self._skills.copy()

    def to_json_list(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._skills.values()]
