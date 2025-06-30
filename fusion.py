import numpy as np
from sklearn.linear_model import LogisticRegression

# Load match scores from both algorithms
match_scores_cosine = np.loadtxt("results/query_match_scores_cosine.csv", delimiter=",")
match_scores_euclidean = np.loadtxt("results/query_match_scores_euclidean.csv", delimiter=",")

# Load subject labels
labels = np.load("results1/dataset_left_labels.npy", allow_pickle=True)  # Adjust for left/right as needed

# Highest Rank Method
def highest_rank_fusion(scores1, scores2):
    """Fusion using the highest rank method."""
    rank1 = np.argsort(-scores1)  # Higher scores get lower ranks
    rank2 = np.argsort(-scores2)

    consensus_rank = np.minimum(rank1, rank2)  # Take the minimum rank
    best_match = labels[np.argmin(consensus_rank)]  # Subject with best (lowest) rank
    return best_match

# Borda Count Method
def borda_count_fusion(scores1, scores2):
    """Fusion using the Borda Count method."""
    rank1 = np.argsort(-scores1)
    rank2 = np.argsort(-scores2)

    consensus_rank = rank1 + rank2  # Sum of ranks
    best_match = labels[np.argmin(consensus_rank)]  # Subject with lowest total rank
    return best_match

# Logistic Regression-Based Fusion
def logistic_regression_fusion(scores1, scores2):
    """Fusion using Logistic Regression."""
    X_train = np.column_stack((scores1, scores2))  # Combine features
    y_train = np.arange(len(scores1))  # Dummy labels (can be actual labels if available)

    model = LogisticRegression()
    model.fit(X_train, y_train)  # Train the model

    fusion_scores = model.predict_proba(X_train)[:, 1]  # Get probability scores
    best_match = labels[np.argmax(fusion_scores)]  # Subject with highest probability
    return best_match

# Apply All Three Fusion Strategies
best_highest_rank = highest_rank_fusion(match_scores_cosine, match_scores_euclidean)
best_borda_count = borda_count_fusion(match_scores_cosine, match_scores_euclidean)
best_logistic_regression = logistic_regression_fusion(match_scores_cosine, match_scores_euclidean)

#Print and Append Results to File
fusion_results = f"""
Fusion Results (New Run):
-------------------------------
Highest Rank Fusion Best Match: {best_highest_rank}
Borda Count Fusion Best Match: {best_borda_count}
Logistic Regression Fusion Best Match: {best_logistic_regression}
"""

print(fusion_results)

# Append results instead of overwriting
with open("results/fusion_results.txt", "a") as f:
    f.write(fusion_results)
