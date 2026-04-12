"""
Generate ensemble comparison figure: Random Forest vs Gradient Boosting decision boundaries.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Dark background color
BG_COLOR = '#1A1A2E'

# Generate data
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# Train models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
gb.fit(X, y)

rf_acc = rf.score(X, y)
gb_acc = gb.score(X, y)

# Create mesh for decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor(BG_COLOR)

models = [rf, gb]
titles = ['Random Forest (Bagging)', 'Gradient Boosting']
accs = [rf_acc, gb_acc]

for ax, model, title, acc in zip(axes, models, titles, accs):
    ax.set_facecolor(BG_COLOR)

    # Decision boundary
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.35, cmap='RdYlGn')

    # Scatter plot
    colors = ['#FF6B6B', '#76FF7A']
    for cls, color in zip([0, 1], colors):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=color, s=25, edgecolors='white',
                   linewidths=0.4, alpha=0.85, zorder=3)

    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel(f'Akurasi: {acc:.1%}', color='#76FF7A', fontsize=10)
    ax.tick_params(colors='gray', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

plt.suptitle('', color='white')
plt.tight_layout(pad=1.5)

output_path = 'figures/ensemble_comparison.pdf'
plt.savefig(output_path, bbox_inches='tight', facecolor=BG_COLOR, dpi=150)
print(f"Saved: {output_path}")
plt.close()
