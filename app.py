import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ga_pima_diabetes import GeneticAlgorithmFeatureSelection, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

# Page Configuration
st.set_page_config(
    page_title="GA Feature Selection Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #050a14;
        background-image: linear-gradient(180deg, #050a14 0%, #001f3f 100%);
    }
    .stApp {
        color: #e6f1ff;
    }
    .stMetric {
        background-color: #0d1b2a;
        padding: 20px;
        border: 1px solid #00d4ff;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
    }
    [data-testid="stSidebar"] {
        background-color: #020c1b;
        border-right: 1px solid #00d4ff33;
    }
    h1, h2, h3 {
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.4);
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #0056b3) !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        padding: 0.5rem 2rem !important;
        transition: 0.3s !important;
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(0, 212, 255, 0.6) !important;
        transform: scale(1.02);
    }
    div[data-testid="stExpander"] {
        background-color: #0a192f;
        border: 1px solid #00d4ff33;
        border-radius: 10px;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.2rem;
        font-weight: bold;
        color: #00d4ff;
    }
    div[data-testid="stRadio"] label p {
        font-size: 1.1rem;
        font-weight: 500;
        color: #00d4ff;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] div {
        padding: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Application Heading
st.title("🧬 Genetic Algorithm: Feature Selection Dashboard")
st.markdown("---")

# Sidebar - Dataset Management
st.sidebar.header("📂 Dataset Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("Custom Dataset Loaded!")
else:
    df = load_data()
    st.sidebar.info("Using Default Pima Indians Dataset")

# Sidebar - GA Hyperparameters
st.sidebar.header("⚙️ GA Hyperparameters")
pop_size = st.sidebar.slider("Population Size", 10, 100, 30)
generations = st.sidebar.slider("Generations", 10, 100, 30)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.20, 0.05)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8)

# Main Dashboard - Configuration Section
st.subheader("🛠️ Genetic Evolution Configuration")

# Dataset Preview & Target Selection (Moved to main for clarity)
with st.expander("📊 1. Dataset Preview & Target Selection", expanded=True):
    st.write(df.head())
    target_col = st.selectbox("Select Target Column", options=df.columns, index=len(df.columns)-1)
    st.info(f"Target: `{target_col}` | Features: {len(df.columns)-1}")

# Method Selection Area (Big and Clear)
st.write("### 🧬 2. Evolutionary Strategy")
method_col1, method_col2, method_col3 = st.columns(3)

with method_col1:
    st.write("**Selection Method**")
    selection_method = st.radio(
        "Who survives?", 
        ["tournament", "roulette", "random", "rank"],
        horizontal=False,
        label_visibility="collapsed"
    )

with method_col2:
    st.write("**Crossover Method**")
    crossover_method = st.radio(
        "Mating Strategy?", 
        ["single", "two", "uniform"],
        horizontal=False,
        label_visibility="collapsed"
    )

with method_col3:
    st.write("**Mutation Method**")
    mutation_method = st.radio(
        "Variation Logic?", 
        ["bitflip", "swap", "inversion"],
        horizontal=False,
        label_visibility="collapsed"
    )

st.markdown("---")

# Main Dashboard Execution Logic
if st.button("🚀 EXECUTE OPTIMIZATION", use_container_width=True):
    
    # 1. Prepare Data
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    feature_names = X.columns

    # 2. Baseline Performance
    st.subheader("🏁 Performance Comparison")
    col1, col2, col3 = st.columns(3)
    
    with st.spinner("Calculating baseline..."):
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y, test_size=0.25, random_state=42)
        baseline_model = RandomForestClassifier(random_state=42)
        baseline_model.fit(X_train_b, y_train_b)
        baseline_acc = accuracy_score(y_test_b, baseline_model.predict(X_test_b))

    # 3. Initialize & Run GA
    ga = GeneticAlgorithmFeatureSelection(
        X.values, y.values,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        selection_method=selection_method,
        crossover_method=crossover_method,
        mutation_method=mutation_method
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Capture outputs or just run
    start_time = time.time()
    
    # Run slightly differently to show progress if possible, 
    # but since ga.run is atomic, we'll use a spinner
    with st.spinner(f"Running GA ({selection_method} + {crossover_method})..."):
        best_mask, best_fitness, hist = ga.run(verbose=False)
        fitness_hist = hist["best_fitness"]
        avg_fitness_hist = hist["avg_fitness"]
        diversity_hist = hist["diversity"]
        feat_hist = hist["feature_count"]
        sel_hist = hist["selection_fitness"]
        cross_hist = hist["crossover_changes"]
        mut_hist = hist["mutation_changes"]
    
    end_time = time.time()
    
    # 4. Final Evaluation
    best_features = [feature_names[i] for i in range(len(best_mask)) if best_mask[i]]
    
    # Final Model Accuracy on Test Set
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(ga.X_train[:, best_mask == 1], ga.y_train)
    test_preds = final_model.predict(ga.X_test[:, best_mask == 1])
    final_test_acc = accuracy_score(ga.y_test, test_preds)

    # 5. Display Main Metrics
    col1.metric("Baseline Accuracy", f"{baseline_acc:.2%}")
    col2.metric("GA Final Accuracy", f"{final_test_acc:.2%}", delta=f"{final_test_acc-baseline_acc:.2%}")
    col3.metric("Execution Time", f"{end_time-start_time:.1f}s")

    st.success(f"🏆 Best Features Found: {', '.join(best_features)}")

    # 6. Comprehensive Visualizations Grid Dashboard
    st.markdown("---")
    st.subheader("📈 GA Optimization Insights Dashboard")
    
    # Prepare extra vars
    improvement = np.diff(fitness_hist)
    
    # Map feature importances back to original feature length
    full_importances = np.zeros(len(feature_names))
    if np.sum(best_mask) > 0:
        full_importances[best_mask == 1] = final_model.feature_importances_
    
    with st.spinner("Compiling Visualizations... Analyzing Feature Frequency..."):
        freq = np.zeros(len(feature_names))
        for _ in range(5): # Fewer runs for responsiveness in Streamlit
            mask, _, _ = ga.run(verbose=False)
            freq += mask
    
    # --- ROW 1 ---
    colA, colB = st.columns(2)
    
    # 1. Fitness vs Diversity
    with colA:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        plt.style.use('dark_background')
        ax1.plot(fitness_hist, color='#00d4ff', linewidth=2, label="Fitness")
        ax1_div = ax1.twinx()
        ax1_div.plot(diversity_hist, color='#ff00ff', linewidth=2, label="Diversity")
        ax1.set_title("Fitness vs Population Diversity", color="#00d4ff")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness SCORE", color='#00d4ff')
        ax1_div.set_ylabel("Avg Hamming Distance", color='#ff00ff')
        ax1.grid(alpha=0.2)
        st.pyplot(fig1)

    # 2. Convergence Speed
    with colB:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(improvement, color='#ffb703', marker='o', markersize=4)
        ax2.set_title("Convergence Speed (Generation-to-Generation Δ)", color="#ffb703")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Improvement Margin")
        ax2.grid(alpha=0.2)
        st.pyplot(fig2)

    # --- ROW 2 ---
    colC, colD = st.columns(2)

    # 3. Model Accuracy Comparison
    with colC:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        bars = ax3.bar(["Baseline Model", "GA Enhanced Model"], [baseline_acc, final_test_acc], color=["#555", "#00d4ff"])
        ax3.set_title("Performance Leap (Accuracy)", color="white")
        ax3.set_ylim(0, 1.0)
        for bar in bars:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2%}", ha='center', va='bottom', color='white', fontweight='bold')
        st.pyplot(fig3)

    # 4. Feature Importance Random Forest
    with colD:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=full_importances, y=list(feature_names), palette="mako", ax=ax4)
        ax4.set_title("Internal RF Feature Importances", color="white")
        st.pyplot(fig4)

    # --- ROW 3 ---
    colE, colF = st.columns(2)

    # 5. GA Feature Frequency Selection
    with colE:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=freq, y=list(feature_names), palette="rocket", ax=ax5)
        ax5.set_title("GA Stability: Feature Selection Frequency (%)", color="white")
        st.pyplot(fig5)

    # 6. Pareto Frontier
    with colF:
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        ax6.scatter(feat_hist, fitness_hist, color="#ff0055", s=80, alpha=0.7, edgecolors='w')
        ax6.set_title("Trade-off: # Features vs Fitness", color="white")
        ax6.set_xlabel("Number of Selected Features")
        ax6.set_ylabel("Validation Fitness Score")
        ax6.grid(alpha=0.2)
        st.pyplot(fig6)

    st.markdown("---")
    st.subheader("🔬 Genetic Operators Dynamics")

    # --- ROW 4 ---
    colG, colH, colI = st.columns(3)

    # 7. Selection Dynamics
    with colG:
        fig7, ax7 = plt.subplots(figsize=(5, 4))
        ax7.plot(avg_fitness_hist, color="#555", linestyle="--", label="Pop Avg Fitness")
        ax7.plot(sel_hist, color="#00ff00", linewidth=2, label="Selected Parents Avg")
        ax7.set_title("Selection Dynamics", color="white")
        ax7.set_xlabel("Generation")
        ax7.set_ylabel("Fitness Score")
        ax7.legend()
        ax7.grid(alpha=0.2)
        st.pyplot(fig7)

    # 8. Crossover Dynamics
    with colH:
        fig8, ax8 = plt.subplots(figsize=(5, 4))
        ax8.plot(cross_hist, color="#00d4ff", marker="o", markersize=4)
        ax8.set_title("Crossover Dynamics", color="white")
        ax8.set_xlabel("Generation")
        ax8.set_ylabel("Avg Bits Changed per Offspring")
        ax8.set_ylim(bottom=0)
        ax8.grid(alpha=0.2)
        st.pyplot(fig8)

    # 9. Mutation Dynamics
    with colI:
        fig9, ax9 = plt.subplots(figsize=(5, 4))
        ax9.plot(mut_hist, color="#ff00ff", marker="s", markersize=4)
        ax9.set_title("Mutation Dynamics", color="white")
        ax9.set_xlabel("Generation")
        ax9.set_ylabel("Avg Bits Mutated per Offspring")
        ax9.set_ylim(bottom=0)
        ax9.grid(alpha=0.2)
        st.pyplot(fig9)

else:
    st.warning("👈 Configure your parameters and click 'EXECUTE OPTIMIZATION' to begin the evolution.")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ for Machine Learning Experimentation")
