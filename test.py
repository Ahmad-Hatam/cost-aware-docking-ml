# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr


import warnings
warnings.filterwarnings('ignore')

# %%
df_orig = pd.read_csv("all_rescoring_results_merged.csv")

# %%
agg_rules = {
    'CNNscore': 'max',
    'CNNaffinity': 'max',
    'smina_affinity': 'max',
    'RTMScore': 'max',
    'SCORCH': 'max',
    'HYDE': 'max',
    'rfscore_v2': 'max', # Assuming rfscores are probabilities/pKd
    'CHEMPLP': 'min',    # Energy/Fitness (lower is better)
    'vina_hydrophobic': 'min',
    'vina_intra_hydrophobic': 'min',
    'true_value': 'first',     # Metadata, just keep the first instance
    'activity_class': 'first'  # Metadata
}

# %%
for col in df_orig.select_dtypes(include=np.number).columns:
    if col not in agg_rules and col not in ["pose", "id", "true_value", "activity_class"]:
        agg_rules[col] = "mean"

df_agg = df_orig.groupby(["id","docking_tool"]).agg(agg_rules).reset_index()
df_agg

# %%
meta_cols = ["true_value", "activity_class"]
score_cols = [c for c in df_agg.columns if c not in meta_cols and c not in ["id","docking_tool"]]
wide = df_agg.set_index(["id", "docking_tool"])[score_cols].unstack("docking_tool")
wide.columns = [f"{tool}_{score}" for score, tool in wide.columns]


meta = df_agg.groupby("id")[meta_cols].first()
df_matrix = wide.join(meta)

df_matrix


# %%
feature_cols = [col for col in df_matrix.columns if col not in meta_cols]
plt.figure(figsize=(15,8))
sns.boxplot(data=df_matrix[feature_cols])
plt.xticks(rotation=90)
plt.title("Distribution of Raw Docking Scores (Checking for Outliers)")
plt.ylabel("Score Value")
plt.tight_layout()
plt.show()

# %%
df_matrix["plants_HYDE"].max(), df_matrix["plants_HYDE"].min(), df_matrix["plants_HYDE"].mean()


# %%

# # Assuming 'df_matrix' is your unstacked dataframe from earlier
# feature_cols = [c for c in df_matrix.columns if c not in ["true_value", "activity_class"]]

# # Create a copy so we don't mess up the original
# df_clean = df_matrix.copy()

# # --- STEP 1: WORST-CASE IMPUTATION ---
# for col in feature_cols:
#     if df_clean[col].isna().any():
#         # Check if this is a 'min' or 'max' metric based on your agg_rules
#         base_metric = col.split('_', 1)[1] # e.g., gets 'CHEMPLP' from 'diffdock_CHEMPLP'
        
#         # If it's an energy score (lower is better), the "worst" is a high positive number
#         if base_metric in ['CHEMPLP', 'vina_hydrophobic', 'vina_intra_hydrophobic']:
#             worst_val = df_clean[col].max()
#             penalty = worst_val + (abs(worst_val) * 0.1) # Make it 10% worse
#         # If it's an affinity/probability score (higher is better), the "worst" is a low number
#         else:
#             worst_val = df_clean[col].min()
#             penalty = worst_val - (abs(worst_val) * 0.1) # Make it 10% worse
            
#         df_clean[col].fillna(penalty, inplace=True)

# # --- STEP 2: IQR CLIPPING (Targeted Outlier Removal) ---
# for col in feature_cols:
#     Q1 = df_clean[col].quantile(0.25)
#     Q3 = df_clean[col].quantile(0.75)
#     IQR = Q3 - Q1
    
#     # 3x IQR is a standard threshold for "extreme" outliers
#     lower_bound = Q1 - 3 * IQR
#     upper_bound = Q3 + 3 * IQR
    
#     df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

# # --- STEP 3: ROBUST SCALING ---
# scaler = RobustScaler()
# df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])

# print("Imputation, Clipping, and Scaling Complete!")

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# 1. DROP NaNs
df_clean = df_matrix.dropna().copy()

print(f"Rows remaining after dropping NaNs: {len(df_clean)}")
feature_cols = [c for c in df_clean.columns if c not in ["true_value", "activity_class"]]
clipping_stats = []

# 2. IQR CLIPPING (To handle HYDE outliers)
for col in feature_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    outliers_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
    
    if outliers_count > 0:
        clipping_stats.append({'Feature': col, 'Values_Squashed': outliers_count})
    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

# 3. ROBUST SCALING
scaler = RobustScaler()
df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])
df_squashed = pd.DataFrame(clipping_stats).sort_values(by='Values_Squashed', ascending=False)
print("Number of extreme values squashed:")
print(df_squashed)

# %%
df_clean.isna().any().unique()

# %%
feature_cols = [col for col in df_clean.columns if col not in meta_cols]
plt.figure(figsize=(15,8))
sns.boxplot(data=df_clean[feature_cols])
plt.xticks(rotation=90)
plt.title("Distribution of Raw Docking Scores (Checking for Outliers)")
plt.ylabel("Score Value")
plt.tight_layout()
plt.show()

# %%


# 1. Calculate Correlations for all features
corr_data = []
for col in feature_cols:
    p_corr, _ = pearsonr(df_clean[col], df_clean['true_value'])
    s_corr, _ = spearmanr(df_clean[col], df_clean['true_value'])
    corr_data.append({
        'Feature': col,
        'Pearson': abs(p_corr),   
        'Spearman': abs(s_corr)
    })

# Convert to DataFrame and sort by the strongest Pearson score
df_corr = pd.DataFrame(corr_data).sort_values(by='Pearson', ascending=False)

# 2. Reshape data for Seaborn (Melt)
df_melted = df_corr.melt(id_vars='Feature', 
                         var_name='Correlation Type', 
                         value_name='Absolute Correlation')

# 3. Create the Visualization
plt.figure(figsize=(12, 16))
sns.set_style("whitegrid")

# Draw the grouped bar chart
ax = sns.barplot(data=df_melted, 
                 y='Feature', 
                 x='Absolute Correlation', 
                 hue='Correlation Type', 
                 palette=['#3498db', '#e74c3c']) 

# Formatting
plt.title("Predictive Strength of All Docking Tools (Pearson vs. Spearman)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Absolute Correlation (|r| and |ρ|)", fontsize=14, fontweight='bold')
plt.ylabel("Docking Tool & Scoring Function", fontsize=14, fontweight='bold')
plt.legend(title='Metric', fontsize=12, title_fontsize=12, loc='lower right')

# Add a vertical threshold line at 0.3
plt.axvline(x=0.3, color='black', linestyle='--', alpha=0.5, label='Moderate Signal Threshold')

plt.tight_layout()
plt.savefig("Full_Feature_Correlation_Analysis.png", dpi=300)
plt.show()

# %%
# 1. Start fresh from the dropped NaNs
df_ml = df_matrix.dropna().copy()

# 2. Define X and y, and SPLIT FIRST (Zero Data Leakage)
X = df_ml[feature_cols].copy()
y = df_ml['true_value'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Strict IQR Clipping (Learn bounds from Train, apply to both)
for col in feature_cols:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)
    X_test[col] = X_test[col].clip(lower=lower_bound, upper=upper_bound)

# 4. Strict Quantile Transformation (Learn shape from Train, apply to both)
qt = PowerTransformer(method='yeo-johnson')
X_train[feature_cols] = qt.fit_transform(X_train[feature_cols])
X_test[feature_cols] = qt.transform(X_test[feature_cols])

# %% [markdown]
# 

# %%
_cost = {
    "localdiffdock": 407.5, "diffdock": 407.5, "flexx": 3.33, "smina": 99.9,
    "gnina": 105.8, "plants": 6.85, "cnnscore": 0.31, "cnnaffinity": 0.31,
    "smina_affinity": 0.31, "ad4": 0.28, "linf9": 0.24, "rtmscore": 0.41,
    "vinardo": 0.29, "scorch": 4.63, "hyde": 2.0, "chemplp": 0.121,
    "rfscore_v1": 0.682, "rfscore_v2": 0.687, "rfscore_v3": 0.69,
    "vina_hydrophobic": 0.69, "vina_intra_hydrophobic": 0.69,
}

for col in feature_cols:
    parts = col.split('_', 1) 
    tool_cost = _cost.get(parts[0].lower(), 0.0)
    score_cost = _cost.get(parts[1].lower(), 0.0) if len(parts) > 1 else 0.0
    total_cost = tool_cost + score_cost if (tool_cost + score_cost) > 0 else 1.0
    
    X_train[col] = X_train[col] / total_cost
    X_test[col] = X_test[col] / total_cost

# 6. Train and Evaluate ML Model (e.g., Lasso)
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)
print(f"🎯 STRICT PIPELINE LASSO R-SQUARED: {r2_score(y_test, y_pred):.4f}")

# %%
# from sklearn.linear_model import Lasso
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score

# # 1. Prepare our X (Features) and y (Target)
# X = df_clean[feature_cols].copy()
# y = df_clean['true_value']

# # 2. Define the exact costs (make sure names match your columns)
# # Note: I'm mapping your original _cost dictionary to our new column names
# cost_dict = {
#     "localdiffdock": 407.5, "diffdock": 407.5, "flexx": 3.33, "smina": 99.9,
#     "gnina": 105.8, "plants": 6.85, "cnnscore": 0.31, "cnnaffinity": 0.31,
#     "smina_affinity": 0.31, "ad4": 0.28, "linf9": 0.24, "rtmscore": 0.41,
#     "vinardo": 0.29, "scorch": 4.63, "hyde": 2.0, "chemplp": 0.121,
#     "rfscore_v1": 0.682, "rfscore_v2": 0.687, "rfscore_v3": 0.69,
#     "vina_hydrophobic": 0.69, "vina_intra_hydrophobic": 0.69
# }

# # Apply the mathematical trick: Divide each column by its tool's cost
# for col in X.columns:
#     tool_name = col.split('_')[0] # Extracts 'diffdock' from 'diffdock_CNNscore'
#     # We add a tiny number (0.01) to avoid dividing by zero if a cost is missing
#     cost = cost_dict.get(tool_name.lower(), 1.0) 
#     X[col] = X[col] / cost

# # 3. Split data into Train and Test sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. Initialize and Train the Lasso Model
# # alpha is our penalty strength. We start with a small penalty.
# lasso_model = Lasso(alpha=0.09, random_state=42)
# lasso_model.fit(X_train, y_train)

# # 5. Evaluate the Model
# y_pred = lasso_model.predict(X_test)
# print(f"R-squared Score: {r2_score(y_test, y_pred):.4f}")
# print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}\n")
# # 6. See which algorithms survived the Cost Pruning!
# weights = pd.DataFrame({
#     'Feature': feature_cols,
#     'Weight': lasso_model.coef_
# })

# # Sort by absolute weight to see the most important features
# weights['Abs_Weight'] = weights['Weight'].abs()
# survivors = weights[weights['Abs_Weight'] > 0].sort_values(by='Abs_Weight', ascending=False)

# print(f"Total features originally: {len(feature_cols)}")
# print(f"Features kept by Cost-Weighted Lasso: {len(survivors)}")
# print("\nThe Pareto Optimal Pipeline Tools (Surviving Features):")
# print(survivors[['Feature', 'Weight']])

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

print("🚀 INITIATING TRANSPARENT COST-AWARE PRUNING 🚀\n")

X_train_rf = X_train.copy()
X_test_rf = X_test.copy()

current_features = list(feature_cols)
history = []

def get_feature_cost(col_name):
    parts = col_name.split('_', 1) 
    tool = parts[0].lower()
    score = parts[1].lower() if len(parts) > 1 else ""
    return _cost.get(tool, 0.0) + _cost.get(score, 0.0)

# The Recursive Pruning Loop
while len(current_features) > 0:
    current_cost = sum([get_feature_cost(f) for f in current_features])
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=4, random_state=42)
    rf.fit(X_train_rf[current_features], y_train)
    
    y_pred = rf.predict(X_test_rf[current_features])
    r2 = r2_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    
    # Calculate Bang-for-Buck early so we know who is getting fired
    importances = rf.feature_importances_
    b4b_scores = []
    
    for idx, feat in enumerate(current_features):
        feat_cost = get_feature_cost(feat)
        if feat_cost == 0: feat_cost = 1.0 
        bang_for_buck = (importances[idx] + 1e-9) / feat_cost
        b4b_scores.append((feat, bang_for_buck))
        
    b4b_scores.sort(key=lambda x: x[1])
    worst_feature = b4b_scores[0][0] # The one about to be fired
    
    # Save the stats WITH NAMES
    history.append({
        'Num_Features': len(current_features),
        'Total_Cost': current_cost,
        'Pearson': pearson_corr,
        'R2_Score': r2,
        'Dropped_Next': worst_feature,
        'Surviving_Features': ", ".join(current_features)
    })
    
    if len(current_features) == 1:
        break 
        
    # Fire the worst feature
    current_features.remove(worst_feature)

df_history = pd.DataFrame(history)

# --- NEW: Print the Cliff Analysis ---
print("==================================================")
print("🔍 THE DIFFDOCK CLIFF ANALYSIS")
print("==================================================")
# Show the steps right around the 20-feature drop
cliff_view = df_history
print(cliff_view[['Num_Features', 'Total_Cost', 'Pearson', 'Dropped_Next']].to_string(index=False))

print("\n==================================================")
print("🧬 THE FINAL 19 CHEAP TOOLS (After DiffDock Dies)")
print("==================================================")
# Show what was left at step 19
tools_left = df_history[df_history['Num_Features'] == 19]['Surviving_Features'].values[0]
for tool in tools_left.split(", "):
    print(f" - {tool}")

for i in history:
    print(f"{i}\n")

# %%
import matplotlib.pyplot as plt
import numpy as np

# 1. Generate Theoretical "Expected" Data
np.random.seed(42)
costs = np.random.uniform(0.1, 500, 30)
accuracies = 0.8 - (0.5 / (np.log10(costs + 1.1))) + np.random.normal(0, 0.05, 30)
accuracies = np.clip(accuracies, 0.2, 0.85)

# Add your specific tools as anchor points
costs = np.append(costs, [407.5, 0.31, 10.5])
accuracies = np.append(accuracies, [0.82, 0.35, 0.78]) # DiffDock, CNNscore, Our Optimal Pipeline

# 2. Setup the Plot
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot all background pipelines
plt.scatter(costs, accuracies, color='grey', alpha=0.6, s=100, label="Sub-optimal Pipelines")

# Plot the expensive baseline
plt.scatter(407.5, 0.82, color='red', alpha=0.8, s=120, edgecolors='black', label="Exhaustive Search (DiffDock)")

# Plot your goal
plt.scatter(10.5, 0.78, color='limegreen', marker='*', s=400, edgecolors='black', label="Optimal Pipeline (Cost-Weighted Lasso)")

# 3. Draw the Pareto Frontier boundary
sorted_indices = np.argsort(costs)
c_sorted = costs[sorted_indices]
a_sorted = accuracies[sorted_indices]
pareto_front = np.maximum.accumulate(a_sorted)
plt.plot(c_sorted, pareto_front, color='red', linestyle='--', linewidth=2, alpha=0.7, label="Pareto Frontier")

# 4. Formatting
plt.xscale('log') 
plt.xlabel("Computational Cost per Molecule (Seconds) [Log Scale]", fontsize=12, fontweight='bold')
plt.ylabel("Expected Predictive Accuracy (Pearson Correlation)", fontsize=12, fontweight='bold')
plt.title("Expected Key Figure: Pareto Optimization of Docking Workflows", fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)

# 5. Save it to your computer!
plt.tight_layout()
plt.savefig("Expected_Key_Figure_Mockup.png", dpi=300)
plt.show()

# %%



