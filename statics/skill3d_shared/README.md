# Skill-3D Shared Memory Skeleton

This directory is the default location for the global Scene Memory and Skill
Library used by Skill-3D.

Only the directory skeleton is tracked. Generated JSON/JSONL memory files and
concrete dynamic skill contents are ignored by git.

Expected generated files include:

- `learned_skills.json`: global skill index
- `questions.json`: question metadata collected during skill evolving
- `memory/episodes/recent.jsonl`: compact episodic memory
- `memory/evidence/facts.jsonl`: compact evidence memory
- `memory/skills/by_id/*.json`: generated skill metadata
- `memory/trajectories/by_class/*.json`: rollout summaries grouped by task
- `progressive_skills/static/**/SKILL.md`: static seed skill definitions
- `progressive_skills/dynamic/**/SKILL.md`: evolved dynamic skill definitions

Use the default runtime settings:

```bash
export MEMORY_ROOT=statics/skill3d_shared
export SKILL3D_SKILL_STORAGE_PATH=${MEMORY_ROOT}/learned_skills.json
export SKILL3D_HIERARCHICAL_MEMORY_DIR=${MEMORY_ROOT}/memory
```
