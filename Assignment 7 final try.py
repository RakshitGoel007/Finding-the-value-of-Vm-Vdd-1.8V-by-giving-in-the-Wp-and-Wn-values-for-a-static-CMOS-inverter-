import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('https://raw.githubusercontent.com/RakshitGoel007/Finding-the-value-of-Vm-Vdd-1.8V-by-giving-in-the-Wp-and-Wn-values-for-a-static-CMOS-inverter-/main/final_Wn_Wp_Vm_TRUE_TRIP_ass7_projB.csv')
# Print columns for verification
print("CSV Columns:", df.columns.tolist())

# Create ratios
df['WN_WP'] = df['WN'] / df['WP']
df['WP_WN'] = df['WP'] / df['WN']

Xbase = df[['WN', 'WP', 'WN_WP', 'WP_WN']].values
y = df['VM'].values

poly = PolynomialFeatures(2, include_bias=False)
X = poly.fit_transform(Xbase)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=1)
model.fit(X_train, y_train)
print("Test R^2 score:", model.score(X_test, y_test))

def predict_vm(wn_um, wp_um):
    # Convert micrometers to meters
    wn = wn_um * 1e-6
    wp = wp_um * 1e-6
    arr = np.array([[wn, wp, wn/wp, wp/wn]])
    arr_poly = poly.transform(arr)
    return model.predict(arr_poly)[0]

# User input *must be in micrometers*
wn_um = float(input("Enter WN in micrometers (e.g. 0.16136): "))
wp_um = float(input("Enter WP in micrometers (e.g. 0.246): "))
vm_pred = predict_vm(wn_um, wp_um)
print(f"Predicted VM for WN={wn_um} µm and WP={wp_um} µm is: {vm_pred}")
