# Lab: Data & Model Versioning with DVC

In this lab, you'll gain hands-on experience with DVC (Data Version Control), a tool for versioning datasets, models, and ML pipelines alongside your code. DVC extends Git to handle large files and creates reproducible ML workflows. By the end of this lab, you'll understand how to track data changes, build ML pipelines, and ensure reproducibility across your team.



## Deliverables

- **Deliverable 1**: Initialize DVC, track a dataset and model with remote storage, and demonstrate switching between versions. Explain to the TA why Git alone is insufficient for ML projects and how DVC solves this problem.

- **Deliverable 2**: Create a multi-stage DVC pipeline  with preprocessing, training, and evaluation stages. Execute the pipeline and show the dependency graph. Explain to the TA how DVC determines which stages need to re-run when you change hyperparameters.

- **Deliverable 3**: Run 3+ experiments with different hyperparameters, compare results, and demonstrate pushing experiments to remote storage. Explain to the TA when you would use DVC versus W&B for experiment tracking, and how they complement each other.

## Getting Started

### Prerequisites
- Git installed
- Python 3.10+

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd mlip-dvc-lab-f25
```

2. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

3. Explore the repository structure:
   - `data/raw/data.csv` - Raw dataset (Breast Cancer Wisconsin dataset with 30 features)
   - `scripts/` - Python scripts for preprocessing, training, and evaluation
   - `params.yaml` - Hyperparameters configuration
   - `dvc.yaml.template` - Template for creating your pipeline
   

## Part 1: Basic DVC Workflow

**Goal**: Learn how DVC tracks large files and enables version switching.

```bash
dvc init
git add .dvcignore
git commit -m "Initialize DVC"
```

**Tasks**:
1. Initialize DVC in your repository
2. Configure remote storage (use local storage for this lab: `/tmp/dvc-storage`)
3. Track `data/raw/data.csv` with DVC
4. Modify the dataset (use `scripts/augment_data.py` to add rows)
5. Track the new version
6. Demonstrate switching between versions using Git and DVC commands
7. Train a model using `scripts/train.py` and track it with DVC

**Hints**:
- Use `dvc --help` to explore available commands
- Examine what files DVC creates when you track a file
- Think about what happens when you commit a `.dvc` file vs. the actual data file
- Explore how `git checkout` and `dvc checkout` work together

## Part 2: DVC Pipelines

**Goal**: Build a reproducible ML pipeline that tracks dependencies and caches results.



**Tasks**:
1. Remove manual tracking from Part 1 (the pipeline will manage outputs)
2. Create a `dvc.yaml` file defining three stages: preprocess, train, and evaluate
3. Define dependencies, parameters, and outputs for each stage
4. Run the pipeline with `dvc repro`
5. Visualize the dependency graph
6. Modify a hyperparameter in `params.yaml` and observe which stages re-run


**Hints**:
- Examine the scripts to understand inputs and outputs
- Check DVC documentation for `dvc.yaml` syntax
- Use `dvc dag` to visualize dependencies
- Experiment with changing files/parameters to understand caching behavior

## Part 3: Experiment Tracking

**Goal**: Run multiple experiments and compare results using DVC's experiment tracking.



**Tasks**:
1. Run at least 3 experiments with different hyperparameters using `dvc exp run`
2. Compare experiment results using `dvc exp show`
3. Apply the best experiment to your workspace
4. Push experiments to remote storage


**Hints**:
- Explore `dvc exp run` options for running experiments
- Make sure metrics are tracked in Git
- Use `dvc exp show` to compare results
- Think about when you would use DVC vs. other experiment tracking tools

As you work through the lab, think about these concepts:

**Pipeline Reproducibility**:
- DVC tracks dependencies (code, data, parameters)
- Only re-runs stages when dependencies change
- `dvc.lock` records exact versions of all dependencies



## Troubleshooting

If you encounter issues:

- **"Cache is empty"**: Use `dvc pull` to retrieve data from remote storage
- **Pipeline won't run**: Check dependencies and file paths in `dvc.yaml`
- **Experiments show `!` for metrics**: Ensure `metrics/scores.json` is tracked in Git
- **Large files in Git**: Remove them and re-add with `dvc add`

For more help, use `dvc <command> --help` or check the [DVC documentation](https://dvc.org/doc).

## Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVC Tutorial](https://dvc.org/doc/start)
- [DVC Pipelines](https://dvc.org/doc/user-guide/pipelines)
- [DVC Experiments](https://dvc.org/doc/user-guide/experiment-management)
