# CMPE 188 Homework 1 – Four New ML Tasks

Apply Linear/Logistic Regression to four new tasks (new datasets, optimization methods, or features). All code is PyTorch and self-verifying via `sys.exit(0)` / `sys.exit(1)`.

## Contents

- **ml_tasks.json** – Full task list including the four new task definitions (pytorch_task_v1).
- **ml_tasks_new_tasks_only.json** – JSON array of only the four new tasks.
- **tasks/** – One folder per new task, each with `task.py`:
  - `linreg_lvl5_diabetes_l1` – Linear regression on Diabetes dataset with L1 (Lasso), AdamW.
  - `linreg_lvl5_lbfgs_optim` – Linear regression with L-BFGS (quasi-Newton) on synthetic data.
  - `logreg_lvl5_wine_multiclass` – Multiclass logistic regression on Wine (3 classes), L2, macro-F1.
  - `logreg_lvl5_scheduler_l1` – Binary logistic regression on Breast Cancer with StepLR and L1 feature selection.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn

```bash
pip install torch numpy scikit-learn
```

## How to run

From the `hw1` directory:

```bash
python tasks/linreg_lvl5_diabetes_l1/task.py
python tasks/linreg_lvl5_lbfgs_optim/task.py
python tasks/logreg_lvl5_wine_multiclass/task.py
python tasks/logreg_lvl5_scheduler_l1/task.py
```

Each script trains, evaluates on train and validation, prints metrics, asserts thresholds, and exits with **0** on success or **1** on failure.

## Submission

Submit either:

1. A link to your **forked** CoderGym repo with these four tasks added under `MLtasks/tasks/` and the updated `ml_tasks.json`, or  
2. A **new** GitHub repo containing only:
   - `ml_tasks.json` (or the four-task snippet)
   - `tasks/linreg_lvl5_diabetes_l1/task.py`
   - `tasks/linreg_lvl5_lbfgs_optim/task.py`
   - `tasks/logreg_lvl5_wine_multiclass/task.py`
   - `tasks/logreg_lvl5_scheduler_l1/task.py`
   - This README (optional).
