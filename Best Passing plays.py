import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.offsetbox as offsetbox

# Load the dataset (Update the file path accordingly)
file_path = "play_by_play_2024 (7).csv"
data = pd.read_csv(file_path)

# Basic data cleaning
data = data.dropna(axis=1, how='all').dropna(axis=0, thresh=50)

# Filter only 'pass' and 'run' plays, exclude special teams
filtered_data = data[data['play_type'].isin(['pass', 'run'])]

# Convert 'xyac_success' to numeric
filtered_data['xyac_success'] = pd.to_numeric(filtered_data['xyac_success'], errors='coerce')

# Drop rows with missing values in the target variable
filtered_data = filtered_data.dropna(subset=['xyac_success'])

# Define features and target for pass plays
pass_data = filtered_data[filtered_data['play_type'] == 'pass']
target = pass_data['xyac_success']
features = pass_data.select_dtypes(include=['float64', 'int64']).drop(columns=['xyac_success', 'play_id']).fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Initialize and fit the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Performance Evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Print RMSE and R² score
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature Importance
feature_importance = pd.Series(rf_model.feature_importances_, index=features.columns).sort_values(ascending=False).head(10)

# Plot Feature Importances
plt.figure(figsize=(12, 6))
feature_importance.plot(kind='bar', color='skyblue')
plt.title("Top 10 Feature Importances for Pass Plays")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# SHAP Analysis
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.show()

# Team-Level Insights
team_predictions = pass_data.copy()
team_predictions['predicted_success'] = rf_model.predict(features)
team_predictions['posteam'] = filtered_data.loc[team_predictions.index, 'posteam']
team_success = team_predictions.groupby('posteam')['predicted_success'].mean().sort_values(ascending=False)

# Display Top 10 Most Successful Teams
top_teams = team_success.head(10)
print("Top 10 Most Successful Teams (Pass Plays):")
print(top_teams)

# Exclude IND and JAX from the top teams
filtered_top_teams = top_teams.drop(['IND', 'JAX'], errors='ignore')

# Load team logos (ensure you have the logo files saved locally with team abbreviations as filenames)
logos = {
    "SF": Image.open("C:/Users/Chuks/Documents/Best Passing Play teams/logos/SF.tif").resize((60, 60)),
    "MIN": Image.open("C:/Users/Chuks/Documents/Best Passing Play teams/logos/MIN.tif").resize((60, 60)),
    "CIN": Image.open("C:/Users/Chuks/Documents/Best Passing Play teams/logos/CIN.tif").resize((60, 60)),
    "ATL": Image.open("C:/Users/Chuks/Documents/Best Passing Play teams/logos/ATL.tif").resize((60, 60)),
    "PHI": Image.open("C:/Users/Chuks/Documents/Best Passing Play teams/logos/PHI.tif").resize((60, 60)),
    "ARI": Image.open("C:/Users/Chuks/Documents/Best Passing Play teams/logos/ARI.tif").resize((60, 60)),
    "GB": Image.open("C:/Users/Chuks/Documents/Best Passing Play teams/logos/GB.tif").resize((60, 60)),
    "BAL": Image.open("C:/Users/Chuks/Documents/Best Passing Play teams/logos/BAL.tif").resize((60, 60))
}

# Plotting the Top Teams by Predicted Pass Play Success Rate (Excluding IND and JAX)
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(filtered_top_teams.index, filtered_top_teams.values,
              color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

# Add team logos on top of the bars
for bar, team in zip(bars, filtered_top_teams.index):
    logo = logos.get(team)
    if logo:
        image_box = offsetbox.OffsetImage(logo, zoom=0.8)
        ab = offsetbox.AnnotationBbox(image_box, (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05),
                                      frameon=False, box_alignment=(0.5, 0))
        ax.add_artist(ab)

# Set plot details
ax.set_title("Top NFL Teams by Predicted Pass Play Success Rate (With Logos)")
ax.set_xlabel("Team")
ax.set_ylabel("Average Predicted Success Rate")
ax.set_ylim(0, 1)
ax.set_xticks([])  # Remove x-axis labels
ax.grid(axis='y')

# Show the final plot
plt.show()
