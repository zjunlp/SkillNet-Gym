"""SkillNet skill-graph construction pipeline.

Stage layout (see the top-level README for full CLI usage):

- ``search``      — semantic search + star-based filtering against SkillNet
- ``download``    — clone candidate skills + LLM quality scoring
- ``dedup``       — embedding-based skill and scenario clustering
- ``scenarios``   — pre/post scenario extraction, alignment, edge review
- ``build_sample``— DAG construction, topology sampling, task evaluation
- ``packaging``   — input-entity evaluation + per-task environment packaging
"""
