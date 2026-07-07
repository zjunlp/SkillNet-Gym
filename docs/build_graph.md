
```bash
# 1. Semantic search over SkillNet for candidate skills per query seed
python -m skillnet_gym.graph.search.skillnet_semantic_search \
    --input query_seeds.json --output skillnet_semantic_results.json \
    --limit 30 --threshold 0.8 --workers 8

# 2. Rank / filter by GitHub stars
python -m skillnet_gym.graph.search.filter_skillnet_results \
    --input skillnet_semantic_results.json \
    --output skillnet_semantic_results_by_stars.json \
    --keep 20 --min-stars 10

# 3. Clone the surviving skill repos
python -m skillnet_gym.graph.download.download_filtered_skills \
    --input skillnet_semantic_results_by_stars.json \
    --target-dir downloaded_skills \
    --manifest downloaded_skills_manifest.json \
    --skip-existing --workers 8

# 4. LLM-scored quality gate (cost / verifiability / documentation)
python -m skillnet_gym.graph.download.evaluate_skills_quality \
    --skills-dir downloaded_skills --workers 8

# 5. Embedding cluster + dedup skills
python -m skillnet_gym.graph.dedup.cluster_dedup_downloaded_skills \
    --manifest downloaded_skills_manifest.json \
    --threshold 0.90 --top-neighbors 50

# 6. Extract pre/post scenarios from each SKILL.md
python -m skillnet_gym.graph.scenarios.extract_skill_scenarios --workers 2

# 7. Dedup scenarios via embedding + Louvain clustering
python -m skillnet_gym.graph.dedup.deduplicate_scenarios \
    --top-neighbors 100 --graph-threshold 0.82 --cluster-threshold 0.88

# 8. Match post-scenarios ↔ pre-scenarios and LLM-verify handoffs
python -m skillnet_gym.graph.scenarios.align_skill_scenarios \
    --top-k 30 --min-retrieval-score 0.5 --workers 8

# 9. Review edges for functional redundancy
python -m skillnet_gym.graph.scenarios.review_skill_edge_redundancy \
    --input scenario_alignment_keep.json --workers 8

# 10. Assemble the directed skill graph
python -m skillnet_gym.graph.build_sample.build_scenario_skill_graph \
    --alignments scenario_alignment_nonredundant_keep.json \
    --output scenario_skill_graph.json

# 11. Sample chain / fan-in / fan-out / diamond DAG tasks
python -m skillnet_gym.graph.build_sample.sample_skill_graph_tasks \
    --max-per-category 1000 --output skill_graph_task_candidates.json

# 12. LLM-review composed tasks for compositional validity
python -m skillnet_gym.graph.build_sample.evaluate_skill_graph_tasks \
    --input skill_graph_task_candidates.json --workers 4

# 13. LLM-score candidate input entities against each task
python -m skillnet_gym.graph.packaging.evaluate_task_input_entities \
    --tasks skill_graph_tasks_part_01.json \
    --entities entity/task_input_entities_part_01.json --workers 4

# 14. Materialize per-task environments (copy skills, download inputs)
python -m skillnet_gym.graph.packaging.package_task_environments \
    --tasks 'skill_graph_tasks_*.json' \
    --entities 'entity/task_input_entities_*.json' \
    --output-dir packaged_tasks --workers 8
```

Every step is checkpoint-friendly — most support `--skip-existing`, `--force`,
and `--workers N`. A one-shot driver is in `scripts/run_graph_pipeline.sh`.