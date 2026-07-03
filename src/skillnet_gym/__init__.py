"""SkillNet-Gym: an open pipeline for constructing a directed skill graph and
auto-synthesizing verifiable multi-skill coding tasks from it.

Two complementary sub-packages:

- :mod:`skillnet_gym.graph` — build a directed skill graph from SkillNet
  search results (search → download → dedup → scenario alignment → graph
  build → task sampling → packaging).

- :mod:`skillnet_gym.synthesis` — DAG-aware task auto-synthesis: takes a
  packaged multi-skill task and produces a verifiable coding task
  (instruction, ``solve.sh``, pytest tests, Dockerfile) by orchestrating
  Claude Code executions.
"""

__version__ = "0.1.0"
