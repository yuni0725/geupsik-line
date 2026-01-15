import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


# Define fitting functions
def linear_func(x, a, b):
    return a * x + b


def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c


def logarithmic_func(x, a, b):
    return a * np.log(x) + b


def polynomial_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def calculate_r_squared(y_actual, y_predicted):
    """Calculate R-squared value"""
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot)


def try_fit_function(func, x_data, y_data, initial_guess, func_name):
    """Try to fit a function and return results"""
    try:
        popt, pcov = curve_fit(func, x_data, y_data, p0=initial_guess, maxfev=5000)
        y_predicted = func(x_data, *popt)
        r_squared = calculate_r_squared(y_data, y_predicted)

        return {
            "name": func_name,
            "function": func,
            "params": popt,
            "r_squared": r_squared,
            "success": True,
        }
    except Exception as e:
        print(f"Failed to fit {func_name}: {e}")
        return {
            "name": func_name,
            "function": func,
            "params": None,
            "r_squared": -1,
            "success": False,
        }


def save_best_parameters(best_fit, filename="best_fit_params.json"):
    """Save best fitted parameters to a JSON file"""
    if best_fit["success"]:
        params = {
            "function_name": best_fit["name"],
            "parameters": [float(p) for p in best_fit["params"]],
            "r_squared": float(best_fit["r_squared"]),
        }

        # Add equation string based on function type
        if best_fit["name"] == "Linear":
            a, b = best_fit["params"]
            params["equation"] = f"y = {a:.3f} * x + {b:.3f}"
        elif best_fit["name"] == "Exponential":
            a, b, c = best_fit["params"]
            params["equation"] = f"y = {a:.3f} * exp({b:.6f} * x) + {c:.3f}"
        elif best_fit["name"] == "Logarithmic":
            a, b = best_fit["params"]
            params["equation"] = f"y = {a:.3f} * ln(x) + {b:.3f}"
        elif best_fit["name"] == "Polynomial":
            a, b, c, d = best_fit["params"]
            params["equation"] = (
                f"y = {a:.6f} * x³ + {b:.3f} * x² + {c:.3f} * x + {d:.3f}"
            )

        with open(filename, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Best fit parameters saved to {filename}")


def create_output_folder():
    """Create output folder with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"fitting_results_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def save_all_parameters(fitting_results, output_folder):
    """Save all fitted parameters to JSON files"""
    all_results = {}
    for result in fitting_results:
        if result["success"]:
            params = {
                "parameters": [float(p) for p in result["params"]],
                "r_squared": float(result["r_squared"]),
            }

            # Add equation string based on function type
            if result["name"] == "Linear":
                a, b = result["params"]
                params["equation"] = f"y = {a:.3f} * x + {b:.3f}"
            elif result["name"] == "Exponential":
                a, b, c = result["params"]
                params["equation"] = f"y = {a:.3f} * exp({b:.6f} * x) + {c:.3f}"
            elif result["name"] == "Logarithmic":
                a, b = result["params"]
                params["equation"] = f"y = {a:.3f} * ln(x) + {b:.3f}"
            elif result["name"] == "Polynomial":
                a, b, c, d = result["params"]
                params["equation"] = (
                    f"y = {a:.6f} * x³ + {b:.3f} * x² + {c:.3f} * x + {d:.3f}"
                )

            all_results[result["name"]] = params

    # Save to JSON file
    filename = os.path.join(output_folder, "all_fitting_results.json")
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"All fitting results saved to {filename}")


def plot_individual_function(x_data, y_data, fit_result, output_folder):
    """Create individual plot for each fitting function"""
    if not fit_result["success"]:
        return

    plt.figure(figsize=(10, 6))

    # Plot original data points
    plt.scatter(
        x_data, y_data, color="red", s=100, alpha=0.7, label="Data Points", zorder=5
    )

    # Generate smooth curve for plotting
    x_smooth = np.linspace(min(x_data), max(x_data), 200)
    y_smooth = fit_result["function"](x_smooth, *fit_result["params"])

    # Create label based on function type
    if fit_result["name"] == "Linear":
        a, b = fit_result["params"]
        fit_label = f"y = {a:.3f}x + {b:.3f}"
        color = "blue"
    elif fit_result["name"] == "Exponential":
        a, b, c = fit_result["params"]
        fit_label = f"y = {a:.1f}*exp({b:.6f}*x) + {c:.1f}"
        color = "green"
    elif fit_result["name"] == "Logarithmic":
        a, b = fit_result["params"]
        fit_label = f"y = {a:.3f}*ln(x) + {b:.3f}"
        color = "orange"
    elif fit_result["name"] == "Polynomial":
        a, b, c, d = fit_result["params"]
        fit_label = f"y = {a:.6f}x³ + {b:.3f}x² + {c:.3f}x + {d:.3f}"
        color = "purple"

    plt.plot(x_smooth, y_smooth, color, linewidth=2, label=fit_label)

    # Customize the plot
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Seconds (sec)", fontsize=12)
    plt.title(
        f"{fit_result['name']} Fit (R² = {fit_result['r_squared']:.6f})",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # Save the plot
    filename = os.path.join(output_folder, f"{fit_result['name'].lower()}_fit.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"{fit_result['name']} plot saved as '{filename}'")


# Read the CSV file
df = pd.read_csv("time.csv")

# Extract data for fitting (remove any rows with zero or negative values for log fitting)
data_filtered = df[(df["sec"] > 0) & (df["x"] > 0)].copy()
x_data = data_filtered["x"].values  # x coordinate as input
y_data = data_filtered["sec"].values  # seconds as output

print(f"Data points: {len(x_data)}")
print(f"X range: {min(x_data):.1f} to {max(x_data):.1f}")
print(f"Y range: {min(y_data):.1f} to {max(y_data):.1f}")

# Try different fitting functions
fitting_results = []

# Linear fit
linear_result = try_fit_function(linear_func, x_data, y_data, [-1, 500], "Linear")
fitting_results.append(linear_result)

# Exponential fit
exp_result = try_fit_function(
    exponential_func, x_data, y_data, [500, -0.01, 0], "Exponential"
)
fitting_results.append(exp_result)

# Logarithmic fit
log_result = try_fit_function(
    logarithmic_func, x_data, y_data, [-100, 600], "Logarithmic"
)
fitting_results.append(log_result)

# Polynomial fit (cubic)
poly_result = try_fit_function(
    polynomial_func, x_data, y_data, [0.001, -0.1, 1, 400], "Polynomial"
)
fitting_results.append(poly_result)

# Find the best fit
successful_fits = [result for result in fitting_results if result["success"]]
if successful_fits:
    best_fit = max(successful_fits, key=lambda x: x["r_squared"])

    print("\n" + "=" * 50)
    print("FITTING RESULTS:")
    print("=" * 50)

    for result in fitting_results:
        if result["success"]:
            print(f"{result['name']:12} | R² = {result['r_squared']:.6f}")
        else:
            print(f"{result['name']:12} | Failed to fit")

    print(f"\nBest fit: {best_fit['name']} with R² = {best_fit['r_squared']:.6f}")

    # Save best parameters
    save_best_parameters(best_fit)

else:
    print("No successful fits found!")
    best_fit = None

# Create output folder with timestamp
output_folder = create_output_folder()
print(f"Created output folder: {output_folder}")

# Save all parameters
save_all_parameters(fitting_results, output_folder)

# Create plots
plt.style.use("default")

# Plot 1: Raw data only
plt.figure(figsize=(10, 6))
plt.scatter(
    x_data, y_data, color="red", s=100, alpha=0.7, label="Data Points", zorder=5
)
plt.xlabel("X Coordinate", fontsize=12)
plt.ylabel("Seconds (sec)", fontsize=12)
plt.title("Seconds vs X Coordinate - Raw Data", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()

# Save to output folder
raw_data_path = os.path.join(output_folder, "raw_data_plot.png")
plt.savefig(raw_data_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Raw data plot saved as '{raw_data_path}'")

# Plot 2: Individual plots for each function
print("\n" + "=" * 50)
print("GENERATING INDIVIDUAL FUNCTION PLOTS:")
print("=" * 50)

for result in fitting_results:
    if result["success"]:
        plot_individual_function(x_data, y_data, result, output_folder)

# Plot 3: All successful fits comparison
successful_fits = [result for result in fitting_results if result["success"]]

if successful_fits:
    plt.figure(figsize=(14, 10))

    # Plot original data points
    plt.scatter(
        x_data,
        y_data,
        color="red",
        s=120,
        alpha=0.8,
        label="Data Points",
        zorder=10,
        edgecolors="darkred",
    )

    # Generate smooth curve for plotting
    x_smooth = np.linspace(min(x_data), max(x_data), 200)

    # Color mapping for different functions
    colors = {
        "Linear": "blue",
        "Exponential": "green",
        "Logarithmic": "orange",
        "Polynomial": "purple",
    }
    linestyles = {
        "Linear": "-",
        "Exponential": "--",
        "Logarithmic": "-.",
        "Polynomial": ":",
    }

    # Plot all successful fits
    for result in successful_fits:
        y_smooth = result["function"](x_smooth, *result["params"])
        color = colors.get(result["name"], "black")
        linestyle = linestyles.get(result["name"], "-")

        plt.plot(
            x_smooth,
            y_smooth,
            color=color,
            linewidth=2.5,
            linestyle=linestyle,
            alpha=0.8,
            label=f"{result['name']} (R² = {result['r_squared']:.6f})",
        )

    # Customize the plot
    plt.xlabel("X Coordinate", fontsize=14)
    plt.ylabel("Seconds (sec)", fontsize=14)
    plt.title("All Function Fits Comparison", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()

    # Save comparison plot
    comparison_path = os.path.join(output_folder, "all_functions_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Comparison plot saved as '{comparison_path}'")

# Display detailed results summary
print(f"\n" + "=" * 60)
print(f"SUMMARY OF ALL RESULTS:")
print(f"=" * 60)

for result in fitting_results:
    if result["success"]:
        print(f"{result['name']:12} | R² = {result['r_squared']:.6f} | Success")
    else:
        print(f"{result['name']:12} | Failed to fit")

if successful_fits:
    best_fit = max(successful_fits, key=lambda x: x["r_squared"])
    print(
        f"\nBest performing function: {best_fit['name']} with R² = {best_fit['r_squared']:.6f}"
    )

    print(f"\n" + "=" * 50)
    print(f"BEST FIT DETAILS:")
    print(f"=" * 50)
    print(f"Function: {best_fit['name']}")
    print(f"R-squared: {best_fit['r_squared']:.6f}")
    print(f"Parameters: {[f'{p:.6f}' for p in best_fit['params']]}")
else:
    print("No successful fits found!")

print(f"\nAll plots and results saved in folder: {output_folder}")
