from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate random dataset X and Y
    X = np.random.uniform(0, 1, N)
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, alpha=0.5, label='Data points')
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red', label='Regression line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Linear Regression: y = {slope:.2f}x + {intercept:.2f}')
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Simulations for slopes and intercepts
    slopes, intercepts = [], []
    for _ in range(S):
        X_sim = np.random.uniform(0, 1, N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Histogram for slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S

    return X, Y, slope, intercept, plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme, slopes, intercepts


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = session.get("N")
    S = session.get("S")
    slope = session.get("slope")
    intercept = session.get("intercept")
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = session.get("beta0")
    beta1 = session.get("beta1")

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # TODO 10: Calculate p-value based on test type
    if test_type == "two_sided":
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= 
                         np.abs(observed_stat - hypothesized_value))
    elif test_type == "greater":
        p_value = np.mean(simulated_stats >= observed_stat)
    else:  # test_type == "less"
        p_value = np.mean(simulated_stats <= observed_stat)

    # TODO 11: If p_value is very small, set fun_message
    if p_value <= 0.0001:
        fun_messages = [
            "Wow! This is rarer than finding a four-leaf clover in a desert! 🍀",
            "That's more significant than a cat willingly taking a bath! 🐱",
            "This is as unlikely as finding a parking spot in downtown at rush hour! 🚗",
            "More surprising than a snowball surviving in a volcano! ❄️🌋"
        ]
        fun_message = np.random.choice(fun_messages)
    else:
        fun_message = None

    # TODO 12: Plot histogram of simulated statistics
    plt.figure(figsize=(10, 6))
    plt.hist(simulated_stats, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(observed_stat, color='red', linestyle='dashed', 
                label=f'Observed {parameter}')
    plt.axvline(hypothesized_value, color='green', linestyle='dashed', 
                label=f'Hypothesized {parameter}')
    
    # Add shaded regions based on test type
    if test_type == "two_sided":
        # Shade both tails for two-sided test
        diff = abs(observed_stat - hypothesized_value)
        extreme_values = simulated_stats[
            abs(simulated_stats - hypothesized_value) >= diff
        ]
        plt.hist(extreme_values, bins=30, color='red', alpha=0.3, 
                edgecolor='black')
    elif test_type == "greater":
        # Shade upper tail
        extreme_values = simulated_stats[simulated_stats >= observed_stat]
        plt.hist(extreme_values, bins=30, color='red', alpha=0.3, 
                edgecolor='black')
    else:  # test_type == "less"
        # Shade lower tail
        extreme_values = simulated_stats[simulated_stats <= observed_stat]
        plt.hist(extreme_values, bins=30, color='red', alpha=0.3, 
                edgecolor='black')

    plt.title(f'Distribution of Simulated {parameter.capitalize()}s\n'
              f'p-value = {p_value:.4f}')
    plt.xlabel(parameter.capitalize())
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot3_path = "static/plot3.png"
    plt.tight_layout()
    plt.savefig(plot3_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Convert confidence level from percentage to proportion
    confidence_level /= 100

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # TODO 14: Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)  # Using n-1 for sample standard deviation

    # TODO 15: Calculate confidence interval using t-distribution
    ci_lower, ci_upper = stats.t.interval(confidence_level, df=S-1, loc=mean_estimate, scale=std_estimate / np.sqrt(S))

    # TODO 16: Check if confidence interval includes true parameter
    includes_true = (ci_lower <= true_param <= ci_upper)

    # TODO 17: Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot individual estimates as small gray points
    plt.scatter(estimates, np.zeros_like(estimates) + 0.1, 
               alpha=0.1, color='gray', s=20, label='Individual estimates')
    
    # Plot mean estimate with color based on whether CI includes true parameter
    point_color = 'green' if includes_true else 'red'
    plt.scatter(mean_estimate, 0.1, color=point_color, s=100, 
               label='Mean estimate', zorder=5)
    
    # Plot confidence interval
    plt.hlines(y=0.1, xmin=ci_lower, xmax=ci_upper, 
              color=point_color, linewidth=2, 
              label=f'{confidence_level*100}% Confidence Interval')
    
    # Plot true parameter value
    plt.axvline(x=true_param, color='blue', linestyle='--', 
               label='True parameter value', zorder=4)
    
    # Add labels and title
    plt.title(f'Confidence Interval for {parameter.capitalize()}\n' + 
             f'({confidence_level*100}% Confidence Level)')
    plt.xlabel(f'{parameter.capitalize()} Value')
    
    # Remove y-axis ticks since they're not meaningful
    plt.yticks([])
    
    # Add annotations for CI bounds and mean
    plt.annotate(f'Lower: {ci_lower:.3f}', xy=(ci_lower, 0.15), 
                xytext=(ci_lower, 0.2), ha='center',
                arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(f'Upper: {ci_upper:.3f}', xy=(ci_upper, 0.15), 
                xytext=(ci_upper, 0.2), ha='center',
                arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(f'Mean: {mean_estimate:.3f}', xy=(mean_estimate, 0.15), 
                xytext=(mean_estimate, 0.2), ha='center',
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Add status message
    status_msg = "CI includes true parameter" if includes_true else "CI does not include true parameter"
    plt.text(0.02, 0.98, status_msg, transform=plt.gca().transAxes, 
            color=point_color, fontsize=10, verticalalignment='top')
    
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plot4_path = "static/plot4.png"
    plt.tight_layout()
    plt.savefig(plot4_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )


if __name__ == "__main__":
    app.run(debug=True)
