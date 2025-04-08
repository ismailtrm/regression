import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations


class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        """Load data from CSV file"""
        self.data = pd.read_csv(self.file_path)
        return self

    def prepare_data(self, feature_column, target_column):
        """Prepare data for regression"""
        self.X = self.data[[feature_column]]
        self.y = self.data[target_column]
        return self


class SalaryPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.y_pred = None
        self.mse = None
        self.r2 = None
        self.intercept = None
        self.coefficient = None

    def train(self, X, y):
        """Train the linear regression model"""
        self.model.fit(X, y)
        self.y_pred = self.model.predict(X)
        self.mse = mean_squared_error(y, self.y_pred)
        self.r2 = r2_score(y, self.y_pred)
        self.intercept = self.model.intercept_
        self.coefficient = self.model.coef_[0]
        return self

    def get_metrics(self):
        """Return model metrics"""
        return {
            'mse': self.mse,
            'r2': self.r2,
            'equation': f"y = {self.intercept:.2f} + {self.coefficient:.2f}x"
        }

    def get_line_equation(self):
        """Return the slope and intercept of the regression line"""
        return self.coefficient, self.intercept


class Visualizer:
    def __init__(self, figsize=(12, 7)):
        self.figsize = figsize
        # Define colors for different levels
        self.colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    def plot_scatter(self, x, y, title):
        """Create scatter plot"""
        plt.figure(figsize=self.figsize)
        plt.scatter(x, y, color="red", alpha=0.5)
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_regression(self, x, y, y_pred, metrics):
        """Create scatter plot with regression line and metrics"""
        plt.figure(figsize=self.figsize)
        plt.scatter(x, y, color="red", alpha=0.5)
        plt.plot(x, y_pred, color="blue", linewidth=2)
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.title("Salary vs. Years of Experience with Linear Regression")

        # Add regression equation and metrics as text
        annotation_text = f"{metrics['equation']}\nMSE: {metrics['mse']:.2f}, R²: {metrics['r2']:.2f}"
        plt.annotate(annotation_text,
                    xy=(0.05, 0.95),
                    xycoords="axes fraction",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_connected_midpoints(self, x, y):
        """Create plot with lines connecting dots and their midpoints"""
        plt.figure(figsize=self.figsize)
        
        # Convert to numpy arrays and sort by x values
        x_array = np.array(x)
        y_array = np.array(y)
        sort_idx = np.argsort(x_array)
        x_sorted = x_array[sort_idx]
        y_sorted = y_array[sort_idx]

        # Plot original points and connect them with lines
        plt.scatter(x_sorted, y_sorted, color="red", alpha=0.5, label="Original Points")
        plt.plot(x_sorted, y_sorted, color="gray", alpha=0.3, linestyle="--")

        # Calculate midpoints
        x_mid = (x_sorted[:-1] + x_sorted[1:]) / 2
        y_mid = (y_sorted[:-1] + y_sorted[1:]) / 2

        # Plot midpoints
        plt.scatter(x_mid, y_mid, color="blue", alpha=0.7, label="Midpoints")
        
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.title("Salary Data with Connected Points and Midpoints")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_all_pair_midpoints(self, x, y):
        """Create plot with midpoints between all possible pairs of points"""
        plt.figure(figsize=self.figsize)
        
        # Convert to numpy arrays
        x_array = np.array(x)
        y_array = np.array(y)
        
        # Plot original points
        plt.scatter(x_array, y_array, color="red", alpha=0.7, s=100, label="Original Points")
        
        # Calculate and plot midpoints between all pairs
        x_mids = []
        y_mids = []
        
        # Use combinations to get all possible pairs of points
        for (i1, i2) in combinations(range(len(x_array)), 2):
            # Calculate midpoint
            x_mid = (x_array[i1] + x_array[i2]) / 2
            y_mid = (y_array[i1] + y_array[i2]) / 2
            
            x_mids.append(x_mid)
            y_mids.append(y_mid)
            
            # Draw faint lines connecting original points to their midpoint
            plt.plot([x_array[i1], x_mid, x_array[i2]], 
                    [y_array[i1], y_mid, y_array[i2]], 
                    color='gray', alpha=0.1, linestyle=':')

        # Plot all midpoints
        plt.scatter(x_mids, y_mids, color="blue", alpha=0.5, s=50, label="All Pair Midpoints")
        
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.title("Salary Data with All Possible Midpoints")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Return midpoints for further processing
        return np.array(x_mids), np.array(y_mids)

    def plot_second_level_midpoints(self, x, y):
        """Create plot with two levels of midpoints"""
        plt.figure(figsize=self.figsize)
        
        # Convert to numpy arrays
        x_array = np.array(x)
        y_array = np.array(y)
        
        # Plot original points
        plt.scatter(x_array, y_array, color="red", alpha=0.7, s=100, label="Original Points")
        
        # Get first level midpoints
        x_mids_1, y_mids_1 = [], []
        for (i1, i2) in combinations(range(len(x_array)), 2):
            x_mid = (x_array[i1] + x_array[i2]) / 2
            y_mid = (y_array[i1] + y_array[i2]) / 2
            x_mids_1.append(x_mid)
            y_mids_1.append(y_mid)
        
        # Convert first level midpoints to arrays
        x_mids_1 = np.array(x_mids_1)
        y_mids_1 = np.array(y_mids_1)
        
        # Plot first level midpoints
        plt.scatter(x_mids_1, y_mids_1, color="blue", alpha=0.5, s=50, label="Level 1 Midpoints")
        
        # Calculate and plot second level midpoints (midpoints of midpoints)
        x_mids_2, y_mids_2 = [], []
        for (i1, i2) in combinations(range(len(x_mids_1)), 2):
            x_mid = (x_mids_1[i1] + x_mids_1[i2]) / 2
            y_mid = (y_mids_1[i1] + y_mids_1[i2]) / 2
            x_mids_2.append(x_mid)
            y_mids_2.append(y_mid)
        
        # Plot second level midpoints
        plt.scatter(x_mids_2, y_mids_2, color="green", alpha=0.5, s=25, 
                   label=f"Level 2 Midpoints (n={len(x_mids_2)})")
        
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.title("Salary Data with Two Levels of Midpoints")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_consecutive_second_level_midpoints(self, x, y):
        """Create plot with two levels of midpoints using consecutive points"""
        plt.figure(figsize=self.figsize)
        
        # Convert to numpy arrays and sort by x values
        x_array = np.array(x)
        y_array = np.array(y)
        sort_idx = np.argsort(x_array)
        x_sorted = x_array[sort_idx]
        y_sorted = y_array[sort_idx]

        # Plot original points and connect them
        plt.scatter(x_sorted, y_sorted, color="red", alpha=0.7, s=100, label="Original Points")
        plt.plot(x_sorted, y_sorted, color="gray", alpha=0.3, linestyle="--")

        # Calculate first level midpoints
        x_mid_1 = (x_sorted[:-1] + x_sorted[1:]) / 2
        y_mid_1 = (y_sorted[:-1] + y_sorted[1:]) / 2

        # Plot first level midpoints and connect them
        plt.scatter(x_mid_1, y_mid_1, color="blue", alpha=0.6, s=50, label="Level 1 Midpoints")
        plt.plot(x_mid_1, y_mid_1, color="blue", alpha=0.3, linestyle="--")

        # Calculate second level midpoints
        x_mid_2 = (x_mid_1[:-1] + x_mid_1[1:]) / 2
        y_mid_2 = (y_mid_1[:-1] + y_mid_1[1:]) / 2

        # Plot second level midpoints
        plt.scatter(x_mid_2, y_mid_2, color="green", alpha=0.7, s=25, 
                   label=f"Level 2 Midpoints (n={len(x_mid_2)})")
        
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.title("Salary Data with Two Levels of Consecutive Midpoints")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_all_pairs_second_level_midpoints(self, x, y):
        """Create plot with two levels of midpoints using all possible pairs"""
        plt.figure(figsize=self.figsize)
        
        # Convert to numpy arrays
        x_array = np.array(x)
        y_array = np.array(y)
        
        # Plot original points
        plt.scatter(x_array, y_array, color="red", alpha=0.7, s=100, label="Original Points")
        
        # Calculate first level midpoints
        x_mids_1, y_mids_1 = [], []
        for (i1, i2) in combinations(range(len(x_array)), 2):
            x_mid = (x_array[i1] + x_array[i2]) / 2
            y_mid = (y_array[i1] + y_array[i2]) / 2
            x_mids_1.append(x_mid)
            y_mids_1.append(y_mid)
        
        x_mids_1 = np.array(x_mids_1)
        y_mids_1 = np.array(y_mids_1)
        
        # Plot first level midpoints
        plt.scatter(x_mids_1, y_mids_1, color="blue", alpha=0.5, s=50, 
                   label=f"Level 1 Midpoints (n={len(x_mids_1)})")
        
        # Calculate second level midpoints
        x_mids_2, y_mids_2 = [], []
        for (i1, i2) in combinations(range(len(x_mids_1)), 2):
            x_mid = (x_mids_1[i1] + x_mids_1[i2]) / 2
            y_mid = (y_mids_1[i1] + y_mids_1[i2]) / 2
            x_mids_2.append(x_mid)
            y_mids_2.append(y_mid)
        
        # Plot second level midpoints
        plt.scatter(x_mids_2, y_mids_2, color="green", alpha=0.5, s=25, 
                   label=f"Level 2 Midpoints (n={len(x_mids_2)})")
        
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.title("Salary Data with Two Levels of All-Pairs Midpoints")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_recursive_midpoints_with_regression(self, x, y, predictor):
        """Create plot with recursive midpoints and compare final line with regression"""
        plt.figure(figsize=self.figsize)
        
        # Convert to numpy arrays and sort by x values
        x_array = np.array(x)
        y_array = np.array(y)
        sort_idx = np.argsort(x_array)
        x_sorted = x_array[sort_idx]
        y_sorted = y_array[sort_idx]

        # Initialize lists to store all levels of midpoints
        all_x_points = [x_sorted]
        all_y_points = [y_sorted]
        
        # Calculate midpoints recursively until only 2 points remain
        current_x = x_sorted
        current_y = y_sorted
        
        while len(current_x) > 2:
            # Calculate midpoints for current level
            x_mid = (current_x[:-1] + current_x[1:]) / 2
            y_mid = (current_y[:-1] + current_y[1:]) / 2
            
            # Store this level's midpoints
            all_x_points.append(x_mid)
            all_y_points.append(y_mid)
            
            # Update current points for next iteration
            current_x = x_mid
            current_y = y_mid

        # Plot all levels
        for level, (x_points, y_points) in enumerate(zip(all_x_points, all_y_points)):
            color = self.colors[level % len(self.colors)]
            size = 100 * (0.7 ** level)  # Decrease point size for each level
            alpha = 0.7 * (0.9 ** level)  # Slightly decrease transparency for each level
            
            # Plot points
            plt.scatter(x_points, y_points, color=color, alpha=alpha, s=size,
                       label=f"Level {level} (n={len(x_points)})")
            
            # Connect points with lines
            plt.plot(x_points, y_points, color=color, alpha=alpha*0.5, linestyle='--')

        # Plot the final two points line
        plt.plot(current_x, current_y, color='black', linewidth=2.5, 
                label='Final Midpoints Line', linestyle='-')

        # Calculate and plot regression line
        reg_slope, reg_intercept = predictor.get_line_equation()
        x_range = np.array([min(x_array), max(x_array)])
        y_range = reg_slope * x_range + reg_intercept
        plt.plot(x_range, y_range, color='red', linewidth=2.5, 
                label='Regression Line', linestyle='-')

        # Calculate angles and equations for both lines
        final_slope = (current_y[1] - current_y[0]) / (current_x[1] - current_x[0])
        final_intercept = current_y[0] - final_slope * current_x[0]
        
        final_angle = np.degrees(np.arctan(final_slope))
        reg_angle = np.degrees(np.arctan(reg_slope))
        angle_diff = abs(final_angle - reg_angle)

        # Add equations and angle information
        info_text = (
            f"Regression Line: y = {reg_slope:.2f}x + {reg_intercept:.2f}\n"
            f"Final Midpoints Line: y = {final_slope:.2f}x + {final_intercept:.2f}\n"
            f"Angle between lines: {angle_diff:.2f}°"
        )
        plt.annotate(info_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                    va='top', fontsize=9)

        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.title("Salary Data: Recursive Midpoints vs Regression Line")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return current_x, current_y, final_slope, final_intercept


def main():
    # Initialize objects
    data_handler = DataHandler("Salary_Data.csv")
    predictor = SalaryPredictor()
    visualizer = Visualizer()

    # Process data
    data_handler.load_data().prepare_data("YearsExperience", "Salary")

    # Train model
    predictor.train(data_handler.X, data_handler.y)

    # Create visualizations
    visualizer.plot_scatter(
        data_handler.data["YearsExperience"],
        data_handler.data["Salary"],
        "Salary vs. Years of Experience (Scatter Plot)"
    )

    visualizer.plot_regression(
        data_handler.data["YearsExperience"],
        data_handler.data["Salary"],
        predictor.y_pred,
        predictor.get_metrics()
    )

    # Plot connected midpoints
    visualizer.plot_connected_midpoints(
        data_handler.data["YearsExperience"],
        data_handler.data["Salary"]
    )

    # Plot all pair midpoints
    visualizer.plot_all_pair_midpoints(
        data_handler.data["YearsExperience"],
        data_handler.data["Salary"]
    )

    # Plot consecutive second level midpoints
    visualizer.plot_consecutive_second_level_midpoints(
        data_handler.data["YearsExperience"],
        data_handler.data["Salary"]
    )

    # Plot all pairs second level midpoints
    visualizer.plot_all_pairs_second_level_midpoints(
        data_handler.data["YearsExperience"],
        data_handler.data["Salary"]
    )

    # Plot recursive midpoints with regression comparison
    final_x, final_y, final_slope, final_intercept = visualizer.plot_recursive_midpoints_with_regression(
        data_handler.data["YearsExperience"],
        data_handler.data["Salary"],
        predictor
    )

    # Print final analysis
    print("\nFinal Analysis:")
    print("Final two points:")
    for x, y in zip(final_x, final_y):
        print(f"Experience: {x:.2f} years, Salary: ${y:.2f}")
    
    reg_slope, reg_intercept = predictor.get_line_equation()
    print(f"\nRegression Line: y = {reg_slope:.2f}x + {reg_intercept:.2f}")
    print(f"Final Midpoints Line: y = {final_slope:.2f}x + {final_intercept:.2f}")
    angle_diff = abs(np.degrees(np.arctan(final_slope)) - np.degrees(np.arctan(reg_slope)))
    print(f"Angle between lines: {angle_diff:.2f}°")


if __name__ == "__main__":
    main()