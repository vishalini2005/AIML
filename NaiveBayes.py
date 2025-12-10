import pandas as pd
from collections import Counter

data = [
    ["Photo", "Morning", "Yes", "High"],
    ["Video", "Afternoon", "Yes", "High"],
    ["Text", "Evening", "No", "Low"],
    ["Photo", "Afternoon", "Yes", "Medium"],
    ["Video", "Morning", "No", "Medium"],
    ["Text", "Morning", "No", "Low"],
    ["Photo", "Evening", "Yes", "High"],
    ["Video", "Evening", "Yes", "High"],
    ["Photo", "Morning", "No", "Medium"],
    ["Text", "Afternoon", "No", "Low"],
]

df = pd.DataFrame(data, columns=["PostType", "TimeOfDay", "HasImage", "Engagement"])

class_counts = Counter(df["Engagement"])
total_rows = len(df)
priors = {c: class_counts[c] / total_rows for c in class_counts}

print("Priors P(Class):")
for c, p in priors.items():
    print(f"P({c}) = {p:.2f}")
print()

def conditional_prob(feature, value, target_class):
    subset = df[df["Engagement"] == target_class]
    count = len(subset[subset[feature] == value])
    return count / len(subset) if len(subset) > 0 else 0

features = {"PostType": "Photo", "TimeOfDay": "Morning", "HasImage": "Yes"}
classes = df["Engagement"].unique()

print("Conditional Probabilities P(Feature|Class):")
for c in classes:
    for f, v in features.items():
        print(f"P({v}|{c}) = {conditional_prob(f, v, c):.3f}")
    print()

posterior_scores = {}
for c in classes:
    score = priors[c]
    for f, v in features.items():
        score *= conditional_prob(f, v, c)
    posterior_scores[c] = score

total_score = sum(posterior_scores.values())
posterior_probs = {c: posterior_scores[c]/total_score if total_score > 0 else 0
                   for c in classes}

print("Posterior Probabilities P(Class|Features):")
for c, p in posterior_probs.items():
    print(f"P({c}|features) = {p:.3f}")

print("\nPrediction:", max(posterior_probs, key=posterior_probs.get))