import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import json
import os


# Define exponential function
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c


def save_parameters(a, b, c, filename="exponential_params.json"):
    """Save fitted parameters to a JSON file"""
    params = {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "equation": f"y = {a:.3f} * exp({b:.6f} * x) + {c:.3f}",
    }
    with open(filename, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Parameters saved to {filename}")


def load_parameters(filename="exponential_params.json"):
    """Load fitted parameters from a JSON file"""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            params = json.load(f)
        print(f"Parameters loaded from {filename}")
        print(f"Loaded equation: {params['equation']}")
        return params["a"], params["b"], params["c"]
    else:
        print(f"No saved parameters found in {filename}")
        return None, None, None


# Read the CSV file
df = pd.read_csv("time.csv")

# Extract data for fitting
x_data = df["sec"].values
y_data = df["x"].values

# Ask user whether to use saved parameters or refit
use_saved = (
    input("Do you want to use saved parameters? (y/n, default=n): ").lower().strip()
)

if use_saved == "y":
    # Try to load saved parameters
    a_fit, b_fit, c_fit = load_parameters()
    if a_fit is None:
        print("No saved parameters found. Proceeding with curve fitting...")
        use_saved = False

if use_saved != "y":
    # Fit exponential function to the data
    try:
        # Initial guess for parameters [a, b, c]
        initial_guess = [500, -0.001, 50]

        # Perform curve fitting
        popt, pcov = curve_fit(exponential_func, x_data, y_data, p0=initial_guess)
        a_fit, b_fit, c_fit = popt

        print(
            f"Fitted exponential function: y = {a_fit:.3f} * exp({b_fit:.6f} * x) + {c_fit:.3f}"
        )

        # Save the fitted parameters
        save_parameters(a_fit, b_fit, c_fit)

    except Exception as e:
        print(f"Error in curve fitting: {e}")
        a_fit, b_fit, c_fit = None, None, None

# Calculate goodness of fit if we have parameters
if a_fit is not None:
    # Generate smooth curve for plotting
    x_smooth = np.linspace(min(x_data), max(x_data), 100)
    y_smooth = exponential_func(x_smooth, a_fit, b_fit, c_fit)

    # Calculate R-squared for goodness of fit
    y_predicted = exponential_func(x_data, a_fit, b_fit, c_fit)
    ss_res = np.sum((y_data - y_predicted) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"R-squared: {r_squared:.4f}")

# Create the plot
plt.figure(figsize=(12, 8))

# Plot original data points
plt.scatter(
    x_data, y_data, color="red", s=100, alpha=0.7, label="Original Data", zorder=5
)

# Plot fitted exponential curve if fitting was successful
if a_fit is not None:
    plt.plot(
        x_smooth,
        y_smooth,
        "blue",
        linewidth=2,
        label=f"Exponential Fit: y = {a_fit:.1f}*exp({b_fit:.6f}*x) + {c_fit:.1f}",
    )

# Customize the plot
plt.xlabel("Seconds (sec)", fontsize=12)
plt.ylabel("X Value", fontsize=12)
plt.title(
    "Exponential Function Fitting to X vs Seconds Data", fontsize=14, fontweight="bold"
)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Add some styling
plt.tight_layout()

# Show the plot
plt.show()

# Optionally save the plot
plt.savefig("exponential_fit_plot.png", dpi=300, bbox_inches="tight")
print("Plot saved as 'exponential_fit_plot.png'")

# Display current parameters for reference
if a_fit is not None:
    print(f"\nCurrent parameters:")
    print(f"a = {a_fit:.6f}")
    print(f"b = {b_fit:.6f}")
    print(f"c = {c_fit:.6f}")
