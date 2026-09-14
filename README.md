===============================================================================
TITANIC SURVIVAL PREDICTION - END-TO-END MACHINE LEARNING PROJECT
===============================================================================

A complete machine learning project built the way a professional would build
it. Every script prints an explanation of what it is doing and why, so running
the project is the lesson.

Problem type : binary classification
Dataset      : 891 records, 12 columns
Target       : Survived (0 = died, 1 = survived)
Baseline     : 0.607  (predict "everyone died")
Result       : 0.801 +/- 0.026 accuracy, measured across 12 stratified splits

-------------------------------------------------------------------------------
CONTENTS
-------------------------------------------------------------------------------

  1. Quick start
  2. Project structure
  3. The eight steps
  4. What the project finds
  5. The most useful thing in here
  6. Concept-to-file map
  7. Ten principles
  8. Where the gains came from
  9. Requirements
 10. Known limitations
 11. Making it your own


-------------------------------------------------------------------------------
1. QUICK START
-------------------------------------------------------------------------------

    pip install -r requirements.txt

    python src/00_generate_data.py    # get the data
    python src/01_eda.py              # explore it
    python src/features.py            # feature engineering (module + demo)
    python src/03_train.py            # compare 8 models
    python src/04_tune.py             # tune, then evaluate ONCE  (5-10 min)
    python src/05_predict.py          # deploy and monitor

Or run everything at once:

    bash run_all.sh

There is also a self-contained Jupyter notebook, titanic_project.ipynb, which
reproduces the whole project in 79 cells with no imports from src/. Upload it
to Google Colab and Run All if you prefer that format.


USING THE REAL KAGGLE DATA
--------------------------
Download train.csv from https://www.kaggle.com/c/titanic/data, drop it into
data/, and step 0 will detect it and leave it alone. Everything else works
unchanged.

Without it, step 0 generates a stand-in dataset that reproduces the real
one's statistical structure: same size, same survival rates by sex and class,
same missing-value pattern, same fare skew, same titles hidden in Name.

See section 10 before you rely on any specific number below.


-------------------------------------------------------------------------------
2. PROJECT STRUCTURE
-------------------------------------------------------------------------------

    titanic_project/
    |
    +-- README.txt
    +-- requirements.txt
    +-- run_all.sh
    +-- titanic_project.ipynb          notebook version, self-contained
    |
    +-- data/
    |   +-- train.csv                  raw data (generated or downloaded)
    |
    +-- src/
    |   +-- 00_generate_data.py        STEP 0  get the data
    |   +-- 01_eda.py                  STEP 1  explore (split FIRST)
    |   +-- features.py                STEP 2  feature engineering - a MODULE
    |   +-- 03_train.py                STEP 3  compare models
    |   +-- 04_tune.py                 STEP 4  tune + final evaluation
    |   +-- 05_predict.py              STEP 5  deploy + monitor
    |
    +-- outputs/
        +-- figures/                   4 charts
        +-- models/
        |   +-- titanic_pipeline.pkl   the deployable artifact
        |   +-- model_card.json        recorded metrics
        +-- model_comparison.csv
        +-- prediction_log.csv

Note that features.py has no number. It is a MODULE imported by the later
steps, not a step you run in sequence. That is deliberate: production must
transform data exactly the way training did, so every transformation lives in
a reusable function rather than a hand-edited notebook cell.

Line counts: 1,644 lines of Python across six files.


-------------------------------------------------------------------------------
3. THE EIGHT STEPS
-------------------------------------------------------------------------------

  STEP           WHAT HAPPENS                      THE POINT
  ----           ------------                      ---------
  0 Frame        define target, metric, baseline   61% baseline - beat it
                                                    or go home
  1 Explore      split, THEN do EDA on train only  snooping bias contaminates
                                                    your brain, not just code
  2 Engineer     Title, FamilySize, HasCabin,      the largest single source
                 log-fare                           of gains
  3 Compare      8 models, cross-validated         breadth first, depth later
  4 Tune         randomized search, test ONCE      training score should FALL
                                                    while CV score RISES
  5 Ship         save whole pipeline, validate,    where real projects
                 monitor                            actually fail


-------------------------------------------------------------------------------
4. WHAT THE PROJECT FINDS
-------------------------------------------------------------------------------

4.1  THE BASELINE
-----------------
Predicting "everybody died" scores 0.607. Any model below that adds nothing.
Quote an accuracy figure without its baseline and the number is meaningless.


4.2  FEATURE ENGINEERING
------------------------
Eleven new features are built, several from columns most people delete.

  FEATURE        BUILT FROM           WHY IT WORKS
  -------        ----------           ------------
  Title          Name                 "Master" meant a boy: 48% survival vs
                                       13% for "Mr". And Name is NEVER
                                       missing, unlike Age (20% blank).
  HasCabin       Cabin being blank    58% vs 32% survival. The missingness
                                       IS a wealth signal, because cabins
                                       were only recorded for upper class.
  FamilySize     SibSp + Parch        each is weak alone; together they show
                                       a clear non-linear pattern
  SexClass       Sex x Pclass         hands the interaction to linear models,
                                       which cannot find it themselves
  FareLog        log1p(Fare)          Fare skew is +4.4; the log tames it

Raw columns: 11. After engineering: 18. As a model matrix: 36 columns.


4.3  THE INTERACTION THAT MATTERS MOST
--------------------------------------
Survival rate by sex and class (training split):

                 1st class   2nd class   3rd class
    female         98.7%       94.9%       59.6%
    male           33.9%        8.0%       14.1%

Class matters far more for women than for men. Tree models find this
automatically; a linear model needs it built by hand.


4.4  MODEL COMPARISON (5-fold cross-validated)
----------------------------------------------
    MODEL                 CV MEAN     STD    TRAIN     GAP
    -----                 -------     ---    -----     ---
    Baseline               0.6067  0.0014   0.6067  0.0000
    LogisticRegression     0.8287  0.0347   0.8413  0.0126
    SVM                    0.8160  0.0353   0.8413  0.0253
    NaiveBayes             0.8034  0.0281   0.7865 -0.0169
    GradientBoosting       0.7992  0.0327   0.9270  0.1277   <- overfit
    RandomForest           0.7936  0.0210   1.0000  0.2064   <- overfit
    KNN                    0.7725  0.0095   0.8610  0.0885
    DecisionTree           0.7627  0.0119   1.0000  0.2373   <- overfit

Two things to read here that beginners skip:

  * THE STANDARD DEVIATION. LogisticRegression beats SVM by 0.013, but both
    have std around 0.035. That difference is smaller than the fold-to-fold
    variation, so it is not yet a real finding.

  * THE GAP, not the training score. A decision tree scoring a perfect 1.000
    on training has not learned anything - it has memorised. The 0.237 gap is
    the alarm.

Also note: one DecisionTree scores 0.763. Averaging 200 of them (RandomForest)
scores 0.794. Their individual mistakes are random and point in different
directions, so they cancel out. That is bagging.


4.5  MORE FEATURES ARE NOT AUTOMATICALLY BETTER
-----------------------------------------------
    MODEL                       RAW    ENGINEERED     DELTA
    -----                       ---    ----------     -----
    LogisticRegression       0.8189       0.8287    +0.0098
    RandomForest (depth 6)   0.8132       0.8202    +0.0070
    RandomForest (no limit)  0.7992       0.7978    -0.0014

Extra features only help a model regularized enough to handle them. Give an
unlimited-depth forest 36 columns instead of 10 and you have handed it more
noise to memorise.

So "add more features" is not good advice on its own. The correct version is:
add BETTER features, and control model complexity to match.

Caveat, and it matters: these deltas are all under 0.01 while the fold-to-fold
std is around 0.02. Re-run with a different random_state and the ranking can
flip. A difference smaller than the variation is not yet a finding - which is
the same lesson as 4.4.


4.6  TUNING IS REGULARIZATION, NOT MAGIC
----------------------------------------
    MODEL                CV BEFORE  CV AFTER   GAP BEFORE  GAP AFTER
    -----                ---------  --------   ----------  ---------
    LogisticRegression      0.8287    0.8301       +0.013     +0.011
    RandomForest            0.7992    0.8202       +0.201     +0.028
    GradientBoosting        0.7992    0.8203       +0.128     +0.006

The forest's training score fell from a perfect 1.000 to 0.848 while its
cross-validated score ROSE. That is exactly what success looks like: a model
that generalizes replacing one that memorized.

We did not make it better at the training data. We made it worse at
memorising, which is what let it generalize.

A model that overfits is not a bad model. It is an UNTUNED model.

Best parameters found (40-iteration randomized search):
    RandomForest      max_depth=8, min_samples_leaf=5, max_features=log2,
                      n_estimators=659
    GradientBoosting  learning_rate=0.011, max_depth=2, n_estimators=162,
                      subsample=0.969
    LogisticRegression  C=1.146


4.7  FINAL EVALUATION (held-out set, used once)
-----------------------------------------------
Winner by cross-validation: LogisticRegression (0.8301)

    Confusion matrix                PREDICTED
                                 died   survived
      ACTUAL  died                 89         20
              survived             27         43

      FP = 20   said survived, actually died    (Type I error)
      FN = 27   said died, actually survived    (Type II error)

      Accuracy   0.7374
      Precision  0.6825    when I said "survived", was I right
      Recall     0.6143    of real survivors, how many I caught
      F1         0.6466
      ROC AUC    0.8135    ranking ability (0.5 = coin flip)


4.8  THE THRESHOLD IS A BUSINESS DECISION
-----------------------------------------
    THRESHOLD  PRECISION  RECALL     F1  ACCURACY
    ---------  ---------  ------     --  --------
          0.3      0.596   0.757  0.667     0.704
          0.4      0.641   0.714  0.676     0.732
          0.5      0.683   0.614  0.647     0.737
          0.6      0.722   0.557  0.629     0.743
          0.7      0.791   0.486  0.602     0.749

Lower threshold flags more people: recall up, precision down. Cancer
screening would go low. A spam filter would go high. 0.5 is a convention,
not a law.


-------------------------------------------------------------------------------
5. THE MOST USEFUL THING IN HERE
-------------------------------------------------------------------------------

Step 4 evaluates on the held-out fifth and gets 0.737, against a
cross-validated 0.830. A 9-point drop.

There are three possible explanations, and you cannot tell them apart from
one number:

  INNOCENT 1 - selection bias. The winner was picked by comparing dozens of
    configurations on CV scores. That selection is itself mild fitting, so
    the CV figure is optimistically biased. Some drop is always expected.

  INNOCENT 2 - an unlucky split. 179 rows is small. One split can hand you an
    easy training set and a hard test set purely by chance.

  GUILTY - leakage, or a genuine failure to generalize.

So instead of shrugging, the script refits the winning configuration across
12 different random splits:

    held-out accuracy over 12 splits
        mean 0.8012    std 0.0258
        min  0.7654    max 0.8436    range 0.0782

    our single split (seed 42) gave 0.7374  ->  0th percentile

VERDICT: seed 42 drew an unlucky split. The held-out number was pessimistic
and the CV number was optimistic. The honest answer is neither - it is
0.801 +/- 0.026, and you only know that because you measured the spread.

A point estimate without a spread is half an answer, whether it is a survey
result or an accuracy score.

WHAT YOU MUST NOT DO: go back and re-tune to make the held-out number look
better. That contaminates it and destroys your only independent estimate.
Re-running across seeds is fine - that measures variance. Tuning against the
held-out set is not.


BUILT-IN VS PERMUTATION IMPORTANCE
----------------------------------
The two disagree, and the gap teaches something. They answer different
questions:

    built-in     "how much did the trees USE this column?"
    permutation  "how much accuracy do I LOSE without it?"

On permutation importance almost everything sits near zero - even Pclass and
Age, which the EDA proved matter. Not because they are useless, but because
they are REDUNDANT: shuffle Pclass and the model still reads class from Fare
and SexClass. Shuffle Age and Title still identifies the children. Only Sex
is irreplaceable.

Low permutation importance means "not uniquely needed", NOT "useless". Never
drop correlated features on that evidence alone.


-------------------------------------------------------------------------------
6. CONCEPT-TO-FILE MAP
-------------------------------------------------------------------------------

    CONCEPT                                   FILE               SECTION
    -------                                   ----               -------
    Split before exploring (snooping bias)    01_eda.py           1.1
    Missingness as a feature                  01_eda.py           1.3
    Target distribution and baseline          01_eda.py           1.4
    Interactions via pivot tables             01_eda.py           1.6
    Skew detection                            01_eda.py           1.7
    Title extraction from text                features.py          -
    Log transform (log1p)                     features.py          -
    Pipelines making leakage impossible       features.py    build_pipeline
    One-hot vs ordinal encoding               features.py          -
    Which models need scaling                 features.py          -
    Establishing a baseline                   03_train.py         3.1
    Cross-validation over single split        03_train.py         3.2
    Reading the standard deviation            03_train.py         3.3
    Train-CV gap as overfitting signal        03_train.py         3.4
    Bagging (1 tree vs 200)                   03_train.py         3.4
    Feature-vs-capacity ablation              03_train.py         3.5
    Feature importance bias                   03_train.py         3.6
    Random vs grid search                     04_tune.py          4.1
    Regularization via hyperparameters        04_tune.py          4.2
    Checking grid edges                       04_tune.py          4.3
    Soft-voting ensembles                     04_tune.py          4.4
    Confusion matrix, precision/recall/F1     04_tune.py          4.5
    Confidence intervals on accuracy          04_tune.py          4.6
    Split variance                            04_tune.py          4.6b
    Threshold as a business decision          04_tune.py          4.7
    Saving the pipeline, not the model        04_tune.py          4.8
    Loading a pipeline for inference          05_predict.py       5.1
    Input validation                          05_predict.py       5.3
    Data drift vs concept drift               05_predict.py       5.4
    Prediction logging                        05_predict.py       5.5
    Presenting to non-technical audiences     05_predict.py       5.6


-------------------------------------------------------------------------------
7. TEN PRINCIPLES
-------------------------------------------------------------------------------

  1. Establish a baseline before anything else. You cannot judge a model
     without one.

  2. Split before you explore. Snooping bias contaminates your judgement,
     not just your code.

  3. Missingness can be a feature. HasCabin gave 58% vs 32%.

  4. Feature engineering beats algorithm choice - but only with matched
     regularization.

  5. Use a Pipeline. It makes leakage structurally impossible, because the
     imputer and scaler refit inside every fold.

  6. Read the standard deviation. Overlapping ranges mean the difference is
     not real.

  7. The train-validation gap is the overfitting diagnostic, not the raw
     training score.

  8. Touch the test set exactly once.

  9. A single split is noisy. Measure the spread before you trust any number.

 10. Deploy the pipeline and monitor it. Models decay; unmonitored models
     fail silently.


-------------------------------------------------------------------------------
8. WHERE THE GAINS CAME FROM
-------------------------------------------------------------------------------

    CHANGE                                GAIN
    ------                                ----
    Baseline -> any real model         +19.4 pts
    Untuned trees -> tuned trees        +2.1 pts
    Raw features -> engineered           +1.0 pts
    Best single model -> ensemble        -0.1 pts

Notice the shape of that table. The unglamorous work - framing the problem,
reading the data, controlling overfitting - produced almost everything. The
ensemble, the most sophisticated step, produced nothing at all.

That ratio is typical of real projects.


-------------------------------------------------------------------------------
9. REQUIREMENTS
-------------------------------------------------------------------------------

Python 3.9 or later, and:

    pandas        >= 2.0
    numpy         >= 1.24
    scikit-learn  >= 1.3
    matplotlib    >= 3.7
    seaborn       >= 0.12
    scipy         >= 1.10
    joblib        >= 1.3

Install with:  pip install -r requirements.txt

Runtime: steps 0, 1, 2, 3 and 5 take seconds each. Step 4 takes 5-10 minutes
because of the hyperparameter search plus the 12-split variance check.

===============================================================================
