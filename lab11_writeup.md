# Lab 11: Data & Model Versioning with DVC

In this lab, you'll gain hands-on experience with DVC (Data Version Control), a tool for versioning datasets, models, and ML pipelines alongside your code. DVC extends Git to handle large files and creates reproducible ML workflows. By the end of this lab, you'll understand how to track data changes, build ML pipelines, and ensure reproducibility across your team.



## Deliverables

- **Deliverable 1**: Initialize DVC, track a dataset and model with remote storage, and demonstrate switching between versions. Explain to the TA why Git alone is insufficient for ML projects and how DVC solves this problem.

- **Deliverable 2**: Create a multi-stage DVC pipeline  with preprocessing, training, and evaluation stages. Execute the pipeline and show the dependency graph. Explain to the TA how DVC determines which stages need to re-run when you change hyperparameters.

- **Deliverable 3**: Run 3+ experiments with different hyperparameters, compare results, and demonstrate pushing experiments to remote storage. Explain to the TA when you would use DVC versus W&B for experiment tracking, and how they complement each other. Additionally, mention how DVC could be used for the team project, specifically with the data stream and other design considerations.

---

## Deliverable 1: Basic DVC Workflow

### Implementation

#### 1. Initialize DVC and Configure Remote Storage

```bash
# Initialize DVC repository
dvc init
git add .dvcignore
git commit -m "Initialize DVC"

# Configure remote storage (local storage for this lab)
dvc remote add -d storage C:\tmp\dvc-storage
```

This creates the `.dvc` directory and configuration files. The remote storage is where DVC will store the actual large files.

#### 2. Track Dataset with DVC

First, we need to remove the file from Git tracking since DVC can't track files already tracked by Git:

```bash
# Remove from Git (but keep file on disk)
git rm -r --cached data/raw/data.csv
git commit -m "stop tracking data/raw/data.csv"

# Add to DVC
dvc add data/raw/data.csv
git add data/raw/data.csv.dvc .gitignore
git commit -m "Track dataset with DVC"
```

**What happens:**
- DVC creates `data/raw/data.csv.dvc` (small metadata file, ~100 bytes)
- DVC stores the actual CSV in its cache
- Git now tracks the `.dvc` file instead of the large CSV
- The CSV is automatically added to `.gitignore`

**File size comparison:**
- Original CSV: ~570 lines, several KB
- `.dvc` file: ~5 lines, ~100 bytes
- Git repository stays small!

#### 3. Modify Dataset and Track New Version

```bash
# Augment dataset (adds 10 new rows)
python scripts/augment_data.py
# Output: Augmented dataset from 569 to 579 samples

# Track the new version
dvc add data/raw/data.csv
git add data/raw/data.csv.dvc
git commit -m "Track augmented dataset version"
```

Now we have two versions tracked:
- **Version 1** (commit `d4cf459`): 569 samples
- **Version 2** (commit `be3b8a0`): 579 samples

#### 4. Demonstrate Version Switching

```bash
# View commit history
git log --oneline
# Shows: be3b8a0 Track augmented dataset version
#        d4cf459 Track dataset with DVC

# Switch to previous version
git checkout d4cf459
dvc checkout

# Verify dataset size
python -c "import pandas as pd; df = pd.read_csv('data/raw/data.csv'); print(f'Dataset has {len(df)} samples')"
# Output: Dataset has 569 samples

# Switch back to latest version
git checkout main
dvc checkout
# Dataset now has 579 samples again
```

**Key insight:** `git checkout` switches the `.dvc` metadata file, and `dvc checkout` downloads the corresponding data file from cache.

#### 5. Train Model and Track with DVC

```bash
# Train model (with hyperparameters: n_estimators=100, max_depth=10)
python scripts/train.py
# Output: Trained model with n_estimators=100, max_depth=10

# Track model with DVC
dvc add models/classifier.pkl
git add models/classifier.pkl.dvc
git commit -m "Track trained model"
```

#### 6. Push to Remote Storage

```bash
dvc push
# Output: 2 files pushed (dataset and model)
```

Files are now stored in `C:\tmp\dvc-storage` and can be retrieved by anyone with access using `dvc pull`.

### Why Git Alone is Insufficient for ML Projects

**Problem 1: File Size Limitations**
- Git is designed for small text files (code, configs)
- ML datasets can be GBs or TBs in size
- Storing large files in Git bloats the repository
- Example: A 1GB dataset would make every clone download 1GB, even if you only need the latest version

**Problem 2: Repository Bloat**
```bash
# Without DVC - if you commit a large dataset:
git add data/raw/data.csv  # 10MB file
git commit -m "Add dataset"
# This 10MB is now permanently in Git history
# Even if you delete it later, it's still in .git/objects
```

**Problem 3: No Version Relationships**
- Git doesn't understand which dataset version was used to train which model
- No way to track: "Model v2.3 was trained on dataset v1.5 with hyperparameters X"
- Hard to reproduce experiments

**Problem 4: Binary File Handling**
- Git stores diffs for text files efficiently
- Binary files (models, images) are stored in full for every version
- A 100MB model file × 10 versions = 1GB in Git history

**Problem 5: Workflow Integration**
- Git doesn't understand ML pipeline dependencies
- Can't automatically detect when to retrain after data changes
- No built-in way to track experiment parameters and results

### How DVC Solves These Problems

**Solution 1: External Storage**
- DVC stores large files outside Git (remote storage: S3, Google Drive, local disk)
- Git only tracks small `.dvc` metadata files (pointers to actual data)
- Repository stays small and fast to clone

**Example:**
```bash
# What Git sees:
$ git ls-files | grep data.csv
data/raw/data.csv.dvc  # Only 100 bytes!

# What DVC tracks:
$ cat data/raw/data.csv.dvc
outs:
- md5: abc123def456...
  size: 245678
  path: data/raw/data.csv
```

**Solution 2: Content-Addressable Storage**
- Files identified by content hash (MD5/SHA)
- Automatic deduplication: same file = same hash = stored once
- Efficient storage even with many versions

**Solution 3: Git Integration**
- Works seamlessly with Git
- `git checkout` switches code and `.dvc` files
- `dvc checkout` syncs actual data files
- Version control for both code AND data

**Demonstration:**
```bash
# Switch to old dataset version
git checkout d4cf459  # Switches .dvc file
dvc checkout          # Downloads corresponding data file
# Now data/raw/data.csv has 569 samples (old version)

# Switch back
git checkout main      # Switches .dvc file back
dvc checkout          # Downloads latest data file
# Now data/raw/data.csv has 579 samples (new version)
```

**Solution 4: Reproducibility**
- `.dvc` files record exact file hashes
- Can reproduce exact dataset/model versions
- `dvc.lock` (in pipelines) records all dependency versions

**Solution 5: Remote Storage**
- Push/pull large files separately from Git
- Team members can `git clone` (fast, small) then `dvc pull` (downloads only needed files)
- Can use cloud storage (S3, GCS) for scalability

**Example workflow:**
```bash
# Team member clones repo (fast, only code)
git clone <repo>
cd repo

# Pulls only the data they need (on-demand)
dvc pull data/raw/data.csv
```

### Key Takeaways

1. **Git tracks metadata, DVC tracks data**: Git stores `.dvc` files (pointers), DVC manages actual files
2. **Version switching**: `git checkout` + `dvc checkout` work together to restore complete project state
3. **Remote storage**: Large files stored separately, keeping Git repo small
4. **Reproducibility**: Exact file versions tracked via content hashes
5. **Team collaboration**: Fast clones, on-demand data downloads

This demonstrates why DVC is essential for ML projects: it extends Git's version control capabilities to handle the large files and complex workflows that Git alone cannot efficiently manage.

---

## Deliverable 2: DVC Pipelines

### Implementation

#### 1. Remove Manual Tracking

Before creating the pipeline, we removed the manual DVC tracking from Part 1, as the pipeline will manage outputs automatically:

```bash
dvc remove data/raw/data.csv.dvc
dvc remove models/classifier.pkl.dvc
git add data/raw/data.csv.dvc models/classifier.pkl.dvc
git commit -m "Remove manual tracking for pipeline"
```

#### 2. Update Configuration Files

**Updated `params.yaml`** to include train hyperparameters:
```yaml
preprocess:
  test_split: 0.2
  random_state: 42

train:
  n_estimators: 100
  max_depth: 10
  random_state: 42
```

**Updated `scripts/train.py`** to load hyperparameters from `params.yaml` and use processed data:
```python
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

n_estimators = params['train']['n_estimators']
max_depth = params['train']['max_depth']
random_state = params['train']['random_state']
data_path = 'data/processed/train.csv'  # Changed from raw data
```

**Updated `scripts/evaluate.py`** to calculate and save metrics:
```python
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1)
}
```

#### 3. Create Multi-Stage Pipeline (`dvc.yaml`)

Created a `dvc.yaml` file defining three stages with their dependencies, parameters, and outputs:

```yaml
stages:
  preprocess:
    cmd: python scripts/preprocess.py
    deps:
      - data/raw/data.csv
      - scripts/preprocess.py
      - params.yaml
    params:
      - preprocess.test_split
      - preprocess.random_state
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python scripts/train.py
    deps:
      - data/processed/train.csv
      - scripts/train.py
      - params.yaml
    params:
      - train.n_estimators
      - train.max_depth
      - train.random_state
    outs:
      - models/classifier.pkl

  evaluate:
    cmd: python scripts/evaluate.py
    deps:
      - models/classifier.pkl
      - data/processed/test.csv
      - scripts/evaluate.py
    metrics:
      - metrics/scores.json:
          cache: false
```

**Key components:**
- **`cmd`**: Command to execute for each stage
- **`deps`**: Input dependencies (files/code that the stage depends on)
- **`params`**: Parameters from `params.yaml` that affect the stage
- **`outs`**: Output files produced by the stage
- **`metrics`**: Metrics files (with `cache: false` so they're always regenerated)

#### 4. Execute the Pipeline

```bash
dvc repro
```

**Output:**
```
Running stage 'preprocess':
> python scripts/preprocess.py
Preprocessed data: train=463, test=116
Generating lock file 'dvc.lock'
Updating lock file 'dvc.lock'

Running stage 'train':
> python scripts/train.py
Trained model with n_estimators=100, max_depth=10
Updating lock file 'dvc.lock'

Running stage 'evaluate':
> python scripts/evaluate.py
Evaluation metrics: {'accuracy': 0.948, 'precision': 0.947, 'recall': 0.973, 'f1': 0.959}
Updating lock file 'dvc.lock'
```

DVC automatically:
- Ran stages in the correct order (preprocess → train → evaluate)
- Tracked all outputs (train.csv, test.csv, classifier.pkl)
- Created `dvc.lock` file recording exact versions of all dependencies

#### 5. Visualize Dependency Graph

```bash
dvc dag
```

**Output:**
```
       +------------+     
       | preprocess |     
       +------------+     
         **        **     
       **            *    
      *               **
+-------+               *
| train |             **
+-------+            *
         **        **
           **    **
             *  *
        +----------+
        | evaluate |
        +----------+
```

This shows:
- `preprocess` runs first (no dependencies on other stages)
- `train` depends on `preprocess` outputs
- `evaluate` depends on both `train` outputs (model) and `preprocess` outputs (test data)

#### 6. Test Dependency Tracking

**Initial state:** All stages up to date
```bash
dvc repro
# Output: Stage 'preprocess' didn't change, skipping
#         Stage 'train' didn't change, skipping
#         Stage 'evaluate' didn't change, skipping
#         Data and pipelines are up to date.
```

**Modified hyperparameter:** Changed `n_estimators` from 100 to 200 in `params.yaml`

```bash
dvc repro
```

**Output:**
```
Running stage 'train':
> python scripts/train.py
Trained model with n_estimators=200, max_depth=10
Updating lock file 'dvc.lock'

Running stage 'evaluate':
> python scripts/evaluate.py
Evaluation metrics: {'accuracy': 0.948, 'precision': 0.947, 'recall': 0.973, 'f1': 0.959}
Updating lock file 'dvc.lock'
```

**Key observation:** `preprocess` was **skipped** because its dependencies didn't change!

### How DVC Determines Which Stages Need to Re-Run

DVC uses a sophisticated dependency tracking system to determine which stages need re-execution:

#### 1. **Content-Based Hashing**
- DVC computes content hashes (MD5) for all dependencies
- Stores these hashes in `dvc.lock` file
- Compares current hashes with stored hashes to detect changes

#### 2. **Dependency Types Tracked**

**File Dependencies (`deps`):**
- If any file in `deps` changes (hash differs), stage must re-run
- Example: If `scripts/train.py` is modified, `train` stage re-runs

**Parameter Dependencies (`params`):**
- Only specific parameters are tracked (e.g., `train.n_estimators`)
- If tracked parameter changes, stage re-runs
- If untracked parameter changes, stage is skipped
- Example: Changing `train.n_estimators` triggers `train`, but changing `preprocess.test_split` does not

**Output Dependencies:**
- If a stage's output is missing or changed, stage re-runs
- Downstream stages automatically re-run if their input dependencies changed

#### 3. **Dependency Graph Traversal**

DVC builds a dependency graph and:
1. Checks each stage's dependencies against `dvc.lock`
2. If any dependency changed → marks stage for re-execution
3. Propagates changes downstream: if Stage A changes, all stages depending on A's outputs must re-run
4. Skips stages with unchanged dependencies (uses cached outputs)

#### 4. **Example: Hyperparameter Change**

When `train.n_estimators` changed from 100 to 200:

```
1. DVC checks preprocess stage:
   - deps: data/raw/data.csv, scripts/preprocess.py, params.yaml
   - params: preprocess.test_split, preprocess.random_state
   - Result: None of these changed → SKIP

2. DVC checks train stage:
   - deps: data/processed/train.csv, scripts/train.py, params.yaml
   - params: train.n_estimators ← CHANGED!
   - Result: Parameter changed → RE-RUN

3. DVC checks evaluate stage:
   - deps: models/classifier.pkl ← This will change (output of train)
   - Result: Input dependency will change → RE-RUN
```

#### 5. **Lock File (`dvc.lock`)**

The `dvc.lock` file records:
- Exact hashes of all dependencies
- Exact parameter values used
- Output file hashes

Example entry:
```yaml
stages:
  train:
    cmd: python scripts/train.py
    deps:
      - path: data/processed/train.csv
        md5: abc123def456...
      - path: params.yaml
        md5: xyz789...
    params:
      params.yaml:
        train.n_estimators: 200  # Current value
        train.max_depth: 10
    outs:
      - path: models/classifier.pkl
        md5: model_hash_here...
```

When you run `dvc repro`, DVC:
1. Reads `dvc.lock` to get baseline
2. Computes current hashes
3. Compares: if different → re-run stage

#### 6. **Benefits of Smart Caching**

- **Efficiency**: Only re-runs what's necessary
- **Reproducibility**: `dvc.lock` ensures exact reproducibility
- **Speed**: Skips expensive operations when dependencies unchanged
- **Correctness**: Automatically handles downstream dependencies

### Key Takeaways

1. **Pipeline Definition**: `dvc.yaml` defines stages, dependencies, and outputs
2. **Automatic Execution**: `dvc repro` runs stages in correct order based on dependencies
3. **Smart Caching**: Stages only re-run when dependencies change
4. **Dependency Tracking**: DVC tracks file hashes and parameter values
5. **Lock File**: `dvc.lock` ensures reproducibility by recording exact dependency versions
6. **Downstream Propagation**: Changes automatically propagate to dependent stages

This demonstrates how DVC creates reproducible ML pipelines that efficiently handle dependency tracking and automatic re-execution, making it easy to iterate on experiments while avoiding unnecessary recomputation.

---

## Deliverable 3: Experiment Tracking

### Implementation

#### 1. Ensure Metrics are Tracked in Git

Before running experiments, we ensured that `metrics/scores.json` is tracked in Git (not DVC), as DVC experiments need metrics files to be in Git for comparison:

```bash
git add metrics/scores.json
git commit -m "Track metrics file for experiments"
```

#### 2. Run Multiple Experiments

We ran 3 experiments with different hyperparameter combinations using `dvc exp run`:

**Experiment 1:**
```bash
dvc exp run -S train.n_estimators=50 -S train.max_depth=5
```
- **Experiment name**: `rival-berk`
- **Results**: accuracy=0.9569, precision=0.9595, recall=0.9726, f1=0.9660

**Experiment 2:**
```bash
dvc exp run -S train.n_estimators=150 -S train.max_depth=15
```
- **Experiment name**: `wrath-coat`
- **Results**: accuracy=0.9569, precision=0.9595, recall=0.9726, f1=0.9660

**Experiment 3:**
```bash
dvc exp run -S train.n_estimators=200 -S train.max_depth=20
```
- **Experiment name**: `dosed-gaze`
- **Results**: accuracy=0.9483, precision=0.9467, recall=0.9726, f1=0.9595

**Key features:**
- The `-S` flag sets parameters temporarily without modifying `params.yaml`
- Each experiment runs the full pipeline (preprocess → train → evaluate)
- DVC automatically tracks parameters, metrics, and code versions
- Experiments are stored as Git commits with special experiment metadata

#### 3. Compare Experiment Results

```bash
dvc exp show
```

**Output:**
```
Experiment                 Created    accuracy   precision   recall        f1   train.n_estimators   train.max_depth
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
workspace                  -           0.94828     0.94667   0.9726   0.95946   200                  20
main                       01:18 PM    0.94828     0.94667   0.9726   0.95946   200                  10
├── 5b56965 [dosed-gaze]   01:18 PM    0.94828     0.94667   0.9726   0.95946   200                  20
├── ea3cb94 [wrath-coat]   01:18 PM     0.9569     0.95946   0.9726   0.96599   150                  15
└── 1137baf [rival-berk]   01:18 PM     0.9569     0.95946   0.9726   0.96599   50                   5
```

**Analysis:**
- **Best performers**: `rival-berk` and `wrath-coat` (tied with f1=0.9660)
- **Interesting finding**: Smaller models (50 estimators, depth 5) performed as well as larger ones (150 estimators, depth 15)
- **Largest model** (200 estimators, depth 20) actually performed slightly worse, suggesting potential overfitting

The table shows:
- All parameters used in each experiment
- All metrics (accuracy, precision, recall, f1)
- File hashes for reproducibility
- Experiment hierarchy (experiments branch from main)

#### 4. Apply Best Experiment

```bash
dvc exp apply rival-berk
```

This applies the best experiment's parameters and model to the workspace, making it the active configuration.

#### 5. Push Experiments to Remote Storage

```bash
dvc exp push origin
```

This pushes experiment data to the Git remote, allowing team members to access and compare experiments. (Note: Authentication may be required depending on remote configuration.)

### DVC vs. W&B (Weights & Biases) for Experiment Tracking

#### When to Use DVC

**Use DVC for experiment tracking when:**

1. **Version Control Integration**: You need experiments tightly integrated with Git version control
   - Experiments are stored as Git commits with metadata
   - Easy to see which code version produced which results
   - Can checkout and reproduce any experiment

2. **Data Versioning**: You need to track which dataset version was used
   - DVC tracks data file versions alongside code
   - Can reproduce exact data + code + parameters combination
   - Essential when datasets change over time

3. **Pipeline Reproducibility**: You want to track entire pipeline executions
   - Records all dependencies (data, code, parameters)
   - `dvc.lock` ensures exact reproducibility
   - Can reproduce entire workflow, not just model training

4. **Local/Private Experiments**: You prefer keeping experiments local or in your own infrastructure
   - No external service required
   - Full control over data and experiments
   - Works offline

5. **Cost Considerations**: You want to avoid cloud service costs
   - Free and open-source
   - No per-user or per-experiment fees
   - Store experiments in your own storage

6. **Simple Workflows**: Your team already uses Git and wants minimal tooling
   - Extends existing Git workflow
   - No additional authentication or setup
   - Familiar interface for developers

#### When to Use W&B

**Use W&B for experiment tracking when:**

1. **Real-time Monitoring**: You need live metrics during training
   - Real-time plots and dashboards
   - Monitor training progress as it happens
   - Compare runs side-by-side in web UI

2. **Rich Visualizations**: You need advanced plotting and analysis
   - Interactive charts and graphs
   - Model architecture visualization
   - Hyperparameter importance analysis

3. **Team Collaboration**: You need a centralized platform for sharing results
   - Web-based interface accessible to all team members
   - Easy sharing and commenting on experiments
   - Built-in collaboration features

4. **Hyperparameter Optimization**: You need automated hyperparameter search
   - Sweeps for hyperparameter tuning
   - Bayesian optimization
   - Automated search strategies

5. **Model Registry**: You need to manage model versions and deployments
   - Model versioning and registry
   - Model deployment tracking
   - Production monitoring integration

6. **Large-Scale Experiments**: You run hundreds or thousands of experiments
   - Better UI for managing many experiments
   - Advanced filtering and search
   - Experiment organization and tagging

#### How They Complement Each Other

**DVC and W&B work together beautifully:**

1. **DVC for Reproducibility, W&B for Analysis**
   - DVC ensures you can reproduce any experiment exactly
   - W&B provides rich visualizations and analysis tools
   - Use DVC to version control, W&B to understand results

2. **DVC for Data, W&B for Metrics**
   - DVC tracks data versions and pipeline dependencies
   - W&B tracks training metrics and visualizations
   - Together: complete picture of data → model → metrics

3. **DVC for Local Development, W&B for Sharing**
   - Use DVC for local experiment tracking and version control
   - Use W&B to share results with team and stakeholders
   - Push important experiments to W&B for visibility

4. **DVC for Pipeline, W&B for Training**
   - DVC manages the full ML pipeline (preprocess → train → evaluate)
   - W&B focuses on the training loop and model metrics
   - DVC ensures reproducibility, W&B provides insights

**Example Workflow:**
```bash
# Use DVC to run experiment with versioned data and code
dvc exp run -S train.n_estimators=100

# Log to W&B during training for real-time monitoring
# (in your training script)
import wandb
wandb.init(project="my-project")
wandb.log({"accuracy": accuracy, "f1": f1})

# DVC tracks the experiment, W&B visualizes it
# Both are valuable for different purposes
```

### DVC for Team Project: Data Stream and Design Considerations

#### Data Stream Management

**1. Streaming Data Versioning**
- **Challenge**: In a streaming data scenario, new data arrives continuously
- **DVC Solution**: 
  - Create periodic snapshots of data streams (e.g., daily/weekly)
  - Use `dvc add` to version each snapshot
  - Tag snapshots with timestamps: `dvc add data/stream_2025-11-19.csv`
  - Track which snapshot was used for each model version

**Example:**
```bash
# Daily data snapshot
dvc add data/stream/2025-11-19.csv
git add data/stream/2025-11-19.csv.dvc
git commit -m "Snapshot: 2025-11-19 stream data"

# Model trained on this snapshot
dvc repro  # Uses data/stream/2025-11-19.csv
```

**2. Incremental Data Updates**
- **Challenge**: Only new data arrives, not full dataset
- **DVC Solution**:
  - Use DVC to track incremental updates
  - Pipeline can process only new data
  - DVC tracks which incremental updates were applied
  - Can reproduce exact data state at any point

**3. Data Quality Tracking**
- **Challenge**: Need to track data quality metrics over time
- **DVC Solution**:
  - Add data quality checks as pipeline stages
  - Track quality metrics in `metrics/` directory
  - Version control quality reports alongside data
  - Can identify when data quality degraded

#### Design Considerations

**1. Pipeline Architecture**

**Modular Stages:**
```yaml
stages:
  ingest:
    cmd: python scripts/ingest_stream.py
    deps:
      - scripts/ingest_stream.py
    outs:
      - data/raw/stream_latest.csv
  
  validate:
    cmd: python scripts/validate_data.py
    deps:
      - data/raw/stream_latest.csv
    metrics:
      - metrics/data_quality.json
  
  preprocess:
    cmd: python scripts/preprocess.py
    deps:
      - data/raw/stream_latest.csv
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
```

**2. Experiment Management for Streaming**

- **Baseline Models**: Use DVC to track baseline models trained on historical data
- **Incremental Training**: Version models trained on incremental data updates
- **A/B Testing**: Use DVC experiments to compare models on same data snapshot
- **Rollback Capability**: Can revert to previous model if new data causes issues

**3. Collaboration Workflow**

**Branch Strategy:**
- Each team member can run experiments on their branch
- DVC experiments are branch-aware
- Merge experiments back to main when validated
- Use `dvc exp push` to share experiments

**Data Sharing:**
- Store data in shared remote storage (S3, GCS, etc.)
- Team members `dvc pull` to get latest data
- DVC handles data synchronization
- No need to manually share large files

**4. Production Deployment**

**Model Versioning:**
- DVC tracks which model version was trained on which data
- `dvc.lock` records exact model + data combination
- Can deploy specific model version with confidence
- Easy rollback if production model fails

**Reproducibility:**
- Production models are fully reproducible
- Can retrain exact same model if needed
- Audit trail of all model versions
- Compliance and regulatory requirements

**5. Monitoring and Alerting**

**Data Drift Detection:**
- Compare current data statistics with training data
- DVC tracks training data statistics
- Pipeline can detect significant drift
- Alert when data distribution changes

**Model Performance Tracking:**
- Track model performance over time
- Compare production metrics with training metrics
- Use DVC to version performance reports
- Identify when model needs retraining

#### Example Team Project Architecture

```
project/
├── data/
│   ├── raw/
│   │   ├── stream_2025-11-19.csv  # Versioned with DVC
│   │   └── stream_2025-11-20.csv
│   └── processed/
│       ├── train.csv  # Pipeline output
│       └── test.csv
├── models/
│   └── classifier.pkl  # Versioned with DVC
├── metrics/
│   ├── scores.json  # Tracked in Git
│   └── data_quality.json
├── dvc.yaml  # Pipeline definition
├── dvc.lock  # Reproducibility lock
└── params.yaml  # Hyperparameters
```

**Workflow:**
1. **Data Ingestion**: New stream data arrives → `dvc add` to version it
2. **Pipeline Execution**: `dvc repro` runs full pipeline
3. **Experiment**: `dvc exp run` tests new hyperparameters
4. **Comparison**: `dvc exp show` compares experiments
5. **Deployment**: Apply best experiment → deploy model
6. **Monitoring**: Track production metrics, retrain when needed

### Key Takeaways

1. **DVC Experiments**: Enable systematic hyperparameter exploration with full reproducibility
2. **Comparison Tools**: `dvc exp show` provides comprehensive experiment comparison
3. **Best Practice Selection**: Easy to identify and apply best-performing configurations
4. **Remote Storage**: Experiments can be shared across team via Git remote
5. **DVC vs W&B**: DVC excels at reproducibility and version control, W&B at visualization and analysis
6. **Complementary Tools**: Use DVC for pipeline management, W&B for training insights
7. **Team Projects**: DVC provides robust data versioning and pipeline management for streaming scenarios
8. **Production Ready**: DVC ensures models are reproducible and auditable in production

This demonstrates how DVC's experiment tracking capabilities enable systematic ML experimentation while maintaining full reproducibility, and how it can be effectively used in team projects with streaming data scenarios.

