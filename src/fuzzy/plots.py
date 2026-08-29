import os
import numpy as np
import matplotlib.pyplot as plt

from src.fuzzy.risk import (
    predicted_score,
    confidence_score,
    priority_score,
    assess_risk
)


# ============================================================
# MELANI - SECTION 2
# FUZZY MEMBERSHIP AND CONTROL-SURFACE PLOTS
# ============================================================


OUTPUT_DIR = os.path.join("docs", "figures")


def ensure_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_predicted_membership():
    plt.figure(figsize=(8, 5))

    for term_name, term in predicted_score.terms.items():
        plt.plot(
            predicted_score.universe,
            term.mf,
            label=term_name
        )

    plt.title("Predicted Score Membership Functions")
    plt.xlabel("Predicted Score")
    plt.ylabel("Membership Degree")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    path = os.path.join(
        OUTPUT_DIR,
        "fuzzy_predicted_membership.png"
    )

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_confidence_membership():
    plt.figure(figsize=(8, 5))

    for term_name, term in confidence_score.terms.items():
        plt.plot(
            confidence_score.universe,
            term.mf,
            label=term_name
        )

    plt.title("Confidence Membership Functions")
    plt.xlabel("Confidence")
    plt.ylabel("Membership Degree")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    path = os.path.join(
        OUTPUT_DIR,
        "fuzzy_confidence_membership.png"
    )

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_priority_membership():
    plt.figure(figsize=(8, 5))

    for term_name, term in priority_score.terms.items():
        plt.plot(
            priority_score.universe,
            term.mf,
            label=term_name
        )

    plt.title("Priority Membership Functions")
    plt.xlabel("Priority")
    plt.ylabel("Membership Degree")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    path = os.path.join(
        OUTPUT_DIR,
        "fuzzy_priority_membership.png"
    )

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_control_surface():
    predicted_values = np.arange(
        0,
        101,
        2,
        dtype=float
    )

    confidence_values = np.arange(
        0,
        10.1,
        0.2,
        dtype=float
    )

    surface = np.zeros(
        (
            len(confidence_values),
            len(predicted_values)
        )
    )

    for i, confidence in enumerate(confidence_values):

        result = assess_risk(
            predicted_values,
            np.full(
                len(predicted_values),
                confidence
            )
        )

        surface[i, :] = result["priority"]

    x_grid, y_grid = np.meshgrid(
        predicted_values,
        confidence_values
    )

    fig = plt.figure(figsize=(10, 7))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.plot_surface(
        x_grid,
        y_grid,
        surface
    )

    ax.set_title(
        "Fuzzy Priority Control Surface"
    )

    ax.set_xlabel(
        "Predicted Score"
    )

    ax.set_ylabel(
        "Confidence"
    )

    ax.set_zlabel(
        "Priority"
    )

    path = os.path.join(
        OUTPUT_DIR,
        "fuzzy_control_surface.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


def generate_all_plots():

    ensure_output_directory()

    print("\nGenerating fuzzy-system figures...\n")

    plot_predicted_membership()
    plot_confidence_membership()
    plot_priority_membership()
    plot_control_surface()

    print("\nAll Member 2 figures generated successfully.")


if __name__ == "__main__":
    generate_all_plots()