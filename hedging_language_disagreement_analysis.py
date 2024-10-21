import pandas as pd
import re
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import classification_report

# Step 1: Load the dataset
file_path = '/data/smk6961/test_set_updated.csv'
df = pd.read_csv(file_path)

# Step 2: Define the specified hedging word list
hedging_words = [
    'may', 'might', 'could', 'can', 'would', 'should', 'indicate', 'suggest', 'tend', 'appear', 'seem',
    'possible', 'likely', 'probable', 'possibly', 'probably', 'perhaps', 'usually', 'generally', 'often',
    'assumption', 'implication', 'uncertain'
]

# Step 3: Function to calculate hedging tokens and types
def calculate_hedging_metrics(text, hedging_words):
    tokens = re.findall(r'\b\w+\b', text.lower())  # Tokenize words
    hedging_tokens = [word for word in tokens if word in hedging_words]  # Match tokens with hedging words
    hedging_types = set(hedging_tokens)  # Unique hedging words
    return len(hedging_tokens), len(hedging_types)

# Step 4: Apply the hedging metrics to the dataset
df['hedge_tokens'], df['hedge_types'] = zip(*df['a_text'].apply(lambda x: calculate_hedging_metrics(x, hedging_words)))

# Step 5: Calculate total word count for normalization
df['total_words'] = df['a_text'].apply(lambda x: len(re.findall(r'\b\w+\b', x.lower())))

# Step 6: Calculate normalized hedging score (tokens + types) / total_words
df['hedging_score'] = (df['hedge_tokens'] + df['hedge_types']) / df['total_words'].replace(0, pd.NA)
df['hedging_score'].fillna(0, inplace=True)  # Fill NaN for cases with zero words

# Step 7: Load sample identifiers from low Jaccard cells (adjust the paths as needed)
low_jaccard_cells = ['T0P1', 'T1P0', 'T1P2']
asrrn_disjoint_ids = []
sfrn_disjoint_ids = []

for cell in low_jaccard_cells:
    asrrn_file = f'/data/smk6961/general_analysis/confusion_matrix_data/{cell}_asrrn_disjoint.csv'
    sfrn_file = f'/data/smk6961/general_analysis/confusion_matrix_data/{cell}_sfrn_disjoint.csv'
    
    # Load AsRRN disjoint sample identifiers
    asrrn_ids = pd.read_csv(asrrn_file)['sample_identifiers'].tolist()
    asrrn_disjoint_ids.extend(asrrn_ids)
    
    # Load SFRN disjoint sample identifiers
    sfrn_ids = pd.read_csv(sfrn_file)['sample_identifiers'].tolist()
    sfrn_disjoint_ids.extend(sfrn_ids)

# Combine and deduplicate identifiers
disagreement_ids = list(set(asrrn_disjoint_ids + sfrn_disjoint_ids))

# Step 8: Filter the dataset to disagreement cases
disagreement_df = df[df['a_id'].isin(disagreement_ids)]

# Step 9: Apply hedging metrics to disagreement cases
disagreement_df['hedge_tokens'], disagreement_df['hedge_types'] = zip(
    *disagreement_df['a_text'].apply(lambda x: calculate_hedging_metrics(x, hedging_words))
)

# Calculate normalized hedging score
disagreement_df['total_words'] = disagreement_df['a_text'].apply(lambda x: len(re.findall(r'\b\w+\b', x.lower())))
disagreement_df['hedging_score'] = (disagreement_df['hedge_tokens'] + disagreement_df['hedge_types']) / disagreement_df['total_words'].replace(0, pd.NA)
disagreement_df['hedging_score'].fillna(0, inplace=True)

# Step 10: Compare hedging levels
print("\nDescriptive Statistics for Hedging Scores in Disagreement Cases:")
print(disagreement_df['hedging_score'].describe())

print("\nHedging Level Distribution in Disagreement Cases:")
print(disagreement_df['hedging_score'].value_counts())

# Step 11: Compare with overall dataset
print("\nDescriptive Statistics for Hedging Scores in Overall Dataset:")
print(df['hedging_score'].describe())

print("\nHedging Level Distribution in Overall Dataset:")
print(df['hedging_score'].value_counts())

# Step 12: Analyze model performance in disagreement cases
def calculate_accuracy(predictions, true_labels):
    correct_predictions = (predictions == true_labels).sum()
    total_answers = len(true_labels)
    return correct_predictions / total_answers if total_answers > 0 else 0

asrrn_accuracy = calculate_accuracy(disagreement_df['AsRRN Prediction'], disagreement_df['True Label'])
sfrn_accuracy = calculate_accuracy(disagreement_df['SFRN Prediction'], disagreement_df['True Label'])

print(f"\nModel Accuracy in Disagreement Cases:")
print(f"AsRRN Accuracy: {asrrn_accuracy:.2%}")
print(f"SFRN Accuracy: {sfrn_accuracy:.2%}")

# Step 13: Perform McNemar's test in disagreement cases (optional)
mcnemar_results = {}
for level in ['No Hedging', 'Low Hedging', 'Medium Hedging', 'High Hedging']:
    subset = disagreement_df[disagreement_df['hedging_level'] == level]

    # Create contingency table for McNemar's test
    asrrn_correct = subset['AsRRN Prediction'] == subset['True Label']
    sfrn_correct = subset['SFRN Prediction'] == subset['True Label']

    # Count outcomes: b (AsRRN correct, SFRN incorrect) and c (SFRN correct, AsRRN incorrect)
    b = ((asrrn_correct == True) & (sfrn_correct == False)).sum()
    c = ((asrrn_correct == False) & (sfrn_correct == True)).sum()

    if b + c > 0:  # Perform McNemar's test only if there are b or c values
        result = mcnemar([[0, b], [c, 0]], exact=True)
        mcnemar_results[level] = {'Statistic': result.statistic, 'p-value': result.pvalue}
        print(f"\nMcNemar's Test for {level} Hedging Level in Disagreement Cases:")
        print(f"Statistic: {result.statistic}, p-value: {result.pvalue}")
    else:
        mcnemar_results[level] = {'Statistic': None, 'p-value': None}
        print(f"\nMcNemar's Test for {level} Hedging Level in Disagreement Cases: Not enough data for test.")

# Step 14: Display Results for Accuracy and McNemar Test
print("\nModel Accuracy by Hedging Level in Disagreement Cases:")
for level, results in accuracy_results.items():
    print(f"{level}: AsRRN Accuracy = {results['AsRRN Accuracy']:.2%}, SFRN Accuracy = {results['SFRN Accuracy']:.2%}, Sample Size = {results['Sample Size']}")

print("\nMcNemar's Test Results for Disagreement Cases (to check significance of model performance difference):")
for level, results in mcnemar_results.items():
    if results['Statistic'] is not None:
        print(f"{level}: McNemar Statistic = {results['Statistic']:.4f}, p-value = {results['p-value']:.4f}")
    else:
        print(f"{level}: Not enough data for McNemar test.")
