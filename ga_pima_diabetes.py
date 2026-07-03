import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')


# ===============================
# Genetic Algorithm Class
# ===============================
class GeneticAlgorithmFeatureSelection:

    def __init__(self, X, y, pop_size=30, generations=50,
                 mutation_rate=0.05, crossover_rate=0.8,
                 selection_method="tournament",
                 crossover_method="single",
                 mutation_method="bitflip",
                 random_state=42):

        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method

        self.n_features = X.shape[1]
        self.random_state = random_state

        # Memoization cache
        self.fitness_cache = {}

        # Train / Validation / Test split (60 / 20 / 20)
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state)
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=self.random_state) # 0.25 * 0.8 = 0.2

        # Scaling
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)

    # ===============================
    # Chromosome
    # ===============================
    def create_chromosome(self):
        chromosome = np.random.randint(0, 2, self.n_features)
        if np.sum(chromosome) == 0:
            chromosome[np.random.randint(0, self.n_features)] = 1
        return chromosome

    # ===============================
    # Fitness Function
    # ===============================
    def fitness(self, chromosome):
        # Memoization check
        chrom_tuple = tuple(chromosome)
        if chrom_tuple in self.fitness_cache:
            return self.fitness_cache[chrom_tuple]

        features = np.where(chromosome == 1)[0]

        if len(features) == 0:
            return 0

        model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        model.fit(self.X_train[:, features], self.y_train)
        
        # USE VALIDATION SET FOR FITNESS
        preds = model.predict(self.X_val[:, features])
        acc = accuracy_score(self.y_val, preds)

        # penalty for many features (penalize if more than 50% are used)
        penalty = 0.01 * (len(features) / self.n_features)
        score = acc - penalty
        
        # Save to cache
        self.fitness_cache[chrom_tuple] = score
        return score

    # ===============================
    # Selection Methods
    # ===============================
    def selection(self, population, fitness_scores):

        if self.selection_method == "tournament":
            i, j = random.sample(range(len(population)), 2)
            return population[i] if fitness_scores[i] > fitness_scores[j] else population[j]

        elif self.selection_method == "roulette":
            fitness_array = np.array(fitness_scores)
            total = fitness_array.sum()
            if total == 0:
                return population[np.random.randint(len(population))]
            probs = fitness_array / total
            idx = np.random.choice(len(population), p=probs)
            return population[idx]

        elif self.selection_method == "random":
            return population[np.random.randint(len(population))]

        elif self.selection_method == "rank":
            ranked = np.argsort(fitness_scores)
            probs = np.linspace(1, 2, len(population))
            probs = probs / probs.sum()
            idx = np.random.choice(ranked, p=probs)
            return population[idx]

    # ===============================
    # Crossover Methods
    # ===============================
    def crossover(self, p1, p2):

        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()

        if self.crossover_method == "single":
            pt = random.randint(1, self.n_features - 1)
            c1 = np.concatenate([p1[:pt], p2[pt:]])
            c2 = np.concatenate([p2[:pt], p1[pt:]])

        elif self.crossover_method == "two":
            pt1 = random.randint(1, self.n_features - 2)
            pt2 = random.randint(pt1, self.n_features - 1)
            c1 = p1.copy()
            c2 = p2.copy()
            c1[pt1:pt2] = p2[pt1:pt2]
            c2[pt1:pt2] = p1[pt1:pt2]

        elif self.crossover_method == "uniform":
            c1 = p1.copy()
            c2 = p2.copy()
            for i in range(self.n_features):
                if random.random() < 0.5:
                    c1[i] = p2[i]
                    c2[i] = p1[i]

        return c1, c2

    # ===============================
    # Mutation Methods
    # ===============================
    def mutation(self, chromosome):

        if self.mutation_method == "bitflip":
            for i in range(self.n_features):
                if random.random() < self.mutation_rate:
                    chromosome[i] = 1 - chromosome[i]

        elif self.mutation_method == "swap":
            i, j = np.random.randint(0, self.n_features, 2)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

        elif self.mutation_method == "inversion":
            i, j = sorted(np.random.randint(0, self.n_features, 2))
            chromosome[i:j] = chromosome[i:j][::-1]

        if np.sum(chromosome) == 0:
            chromosome[np.random.randint(0, self.n_features)] = 1

        return chromosome

    # ===============================
    # Run Genetic Algorithm
    # ===============================
    def run(self, verbose=True):

        population = [self.create_chromosome() for _ in range(self.pop_size)]

        best_chromosome = None
        best_fitness = -1

        history = {
            "best_fitness": [],
            "avg_fitness": [],
            "diversity": [],
            "feature_count": [],
            "selection_fitness": [],
            "crossover_changes": [],
            "mutation_changes": []
        }

        for gen in range(self.generations):

            fitness_scores = [self.fitness(c) for c in population]

            # Best individual
            idx = np.argmax(fitness_scores)
            if fitness_scores[idx] > best_fitness:
                best_fitness = fitness_scores[idx]
                best_chromosome = population[idx].copy()

            history["best_fitness"].append(best_fitness)
            history["avg_fitness"].append(np.mean(fitness_scores))
            history["feature_count"].append(np.sum(best_chromosome))

            # Diversity (Vectorized Hamming Distance calculation)
            pop_array = np.array(population)
            # Efficiently compute average distance between all pairs
            # Matrix: (pop_size, pop_size)
            diff = (pop_array[:, None, :] != pop_array[None, :, :]).sum(axis=2)
            diversity = np.mean(diff[np.triu_indices(self.pop_size, k=1)])
            history["diversity"].append(diversity)

            if verbose and (gen + 1) % 5 == 0:
                print(f"Gen {gen+1}/{self.generations} | Best: {best_fitness:.4f} | Size: {np.sum(best_chromosome)}")

            # New population
            new_population = [best_chromosome]
            
            selected_parents_fitness = []
            gen_crossover_dist = 0
            gen_mutation_count = 0
            crossover_pairs = 0
            mutation_individuals = 0

            while len(new_population) < self.pop_size:
                p1 = self.selection(population, fitness_scores)
                p2 = self.selection(population, fitness_scores)
                
                selected_parents_fitness.extend([self.fitness(p1), self.fitness(p2)])

                c1_pre, c2_pre = self.crossover(p1, p2)
                
                gen_crossover_dist += np.sum(c1_pre != p1) + np.sum(c2_pre != p2)
                crossover_pairs += 2
                
                c1 = c1_pre.copy()
                c2 = c2_pre.copy()

                m1 = self.mutation(c1)
                gen_mutation_count += np.sum(m1 != c1_pre)
                mutation_individuals += 1
                new_population.append(m1)
                
                if len(new_population) < self.pop_size:
                    m2 = self.mutation(c2)
                    gen_mutation_count += np.sum(m2 != c2_pre)
                    mutation_individuals += 1
                    new_population.append(m2)

            population = new_population
            
            history["selection_fitness"].append(np.mean(selected_parents_fitness) if selected_parents_fitness else 0)
            history["crossover_changes"].append(gen_crossover_dist / crossover_pairs if crossover_pairs > 0 else 0)
            history["mutation_changes"].append(gen_mutation_count / mutation_individuals if mutation_individuals > 0 else 0)

        return best_chromosome, best_fitness, history


# ===============================
# Load Pima Dataset
# ===============================
def load_data(file_path=None):
    if file_path is not None:
        df = pd.read_csv(file_path)
        return df
    
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
             'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
    df = pd.read_csv(url, names=names)
    return df


# ===============================
# MAIN
# ===============================
def main():

    df = load_data()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    feature_names = X.columns

    # Baseline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    baseline = accuracy_score(y_test, model.predict(X_test))
    print("Baseline Accuracy:", baseline)

    # Run GA
    ga = GeneticAlgorithmFeatureSelection(
        X.values, y.values,
        selection_method="roulette",
        crossover_method="uniform",
        mutation_method="bitflip",
        generations=30, # Reduced for faster demo
        pop_size=20
    )

    print("\nStarting Genetic Algorithm Evolution...")
    best_mask, best_fitness, hist = ga.run()
    fitness_hist = hist["best_fitness"]
    diversity_hist = hist["diversity"]
    feat_hist = hist["feature_count"]
    avg_fitness_hist = hist["avg_fitness"]
    sel_hist = hist["selection_fitness"]
    cross_hist = hist["crossover_changes"]
    mut_hist = hist["mutation_changes"]

    selected_features = [feature_names[i] for i in range(len(best_mask)) if best_mask[i]]

    # FINAL EVALUATION ON HELD-OUT TEST SET (Unseen by GA)
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(ga.X_train[:, best_mask == 1], ga.y_train)
    test_preds = final_model.predict(ga.X_test[:, best_mask == 1])
    final_test_acc = accuracy_score(ga.y_test, test_preds)

    print("\n" + "="*30)
    print("GA RESULTS")
    print("="*30)
    print(f"Best Validation Fitness: {best_fitness:.4f}")
    print(f"Final Test Accuracy:     {final_test_acc:.4f}")
    print(f"Selected Features ({len(selected_features)}): {selected_features}")
    print("="*30)

    # ===============================
    # Visualizations (Combined Dashboard)
    # ===============================
    
    improvement = np.diff(fitness_hist)
    
    # Map feature importances back to original length
    full_importances = np.zeros(len(feature_names))
    if np.sum(best_mask) > 0:
        full_importances[best_mask == 1] = final_model.feature_importances_

    # Feature frequency (Optimized with memoization)
    print("\nRunning Feature Frequency Analysis (20 runs)...")
    freq = np.zeros(len(feature_names))
    for i in range(20):
        # We reuse the same 'ga' object to benefit from the growing self.fitness_cache
        mask, _, _ = ga.run(verbose=False)
        freq += mask
        if (i+1) % 5 == 0:
            print(f"Completed {i+1}/20 runs...")

    # Create a 4x3 figure to combine all plots
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle("Genetic Algorithm Feature Selection - Optimization Results", fontsize=20, fontweight='bold')

    # 1. Fitness & Diversity
    ax = axes[0, 0]
    ax.plot(fitness_hist, label="Fitness", color="teal")
    ax_div = ax.twinx()
    ax_div.plot(diversity_hist, label="Diversity", color="purple")
    ax.set_title("Fitness & Population Diversity")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax_div.set_ylabel("Diversity")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_div.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="lower right")

    # 2. Convergence Speed
    ax = axes[0, 1]
    ax.plot(improvement, color="orange", marker=".")
    ax.set_title("Fitness Improvement per Generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Improvement (Δ Fitness)")

    # 3. Feature Stability Heatmap
    ax = axes[0, 2]
    sns.heatmap(np.array([best_mask]*10), cbar=False, cmap="Blues", ax=ax)
    ax.set_title("Final Selected Features Stability (Mask)")
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks([])

    # 4. Baseline vs GA Accuracy
    ax = axes[1, 0]
    bars = ax.bar(["Baseline Model", "GA Optimized Model"], [baseline, final_test_acc], color=["gray", "blue"])
    ax.set_title("Performance Comparison")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.0)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2%}", ha='center', va='bottom', fontweight='bold')

    # 5. Feature Importance (Random Forest Internal)
    ax = axes[1, 1]
    ax.bar(feature_names, full_importances, color="teal")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_title("Feature Importance (Random Forest)")

    # 6. Feature Selection Frequency (GA Multiple Runs)
    ax = axes[1, 2]
    ax.bar(feature_names, freq, color="purple")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_title("Feature Frequency Selection (20 Runs)")

    # 7. Pareto: Fitness vs Features
    ax = axes[2, 0]
    ax.scatter(fitness_hist, feat_hist, color="red", alpha=0.6, s=50)
    ax.set_title("Trade-off: Fitness vs Number of Features")
    ax.set_xlabel("Fitness")
    ax.set_ylabel("Number of Features")

    # 8. Selection Dynamics
    ax = axes[2, 1]
    ax.plot(avg_fitness_hist, label="Pop Avg Fitness", color="gray", linestyle="--")
    ax.plot(sel_hist, label="Selected Parents Avg", color="green")
    ax.set_title("Selection Dynamics")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness SCORE")
    ax.legend(loc="lower right")

    # 9. Crossover Dynamics
    ax = axes[2, 2]
    ax.plot(cross_hist, color="blue", marker="o", markersize=4)
    ax.set_title("Crossover Dynamics (Avg Bits Changed)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg Bits Changed per Offspring")
    ax.set_ylim(bottom=0)

    # 10. Mutation Dynamics
    ax = axes[3, 0]
    ax.plot(mut_hist, color="magenta", marker="s", markersize=4)
    ax.set_title("Mutation Dynamics (Avg Bits Mutated)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg Bits Mutated per Offspring")
    ax.set_ylim(bottom=0)

    # Hide unused subplots
    axes[3, 1].axis('off')
    axes[3, 2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.94) # Adjust for the super title
    
    output_filename = "combined_ga_dashboard.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nAll visualizations combined and successfully saved to '{output_filename}'! 📊")


if __name__ == "__main__":
    main()