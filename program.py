from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['53k']
collection = db['77k']

# Query 1: Count charities with more than 10 programs
charities_with_10plus_programs = collection.count_documents({
    "programs": {"$exists": True},
    "$expr": {"$gt": [{"$size": "$programs"}, 10]}
})

print(f"Number of charities with more than 10 programs: {charities_with_10plus_programs}")

# Query 2: Get all charities with their program counts for histogram
pipeline = [
    {
        "$project": {
            "selected_name_final_normalized": 1,
            "program_count": {
                "$cond": {
                    "if": {"$isArray": "$programs"},
                    "then": {"$size": "$programs"},
                    "else": 0
                }
            }
        }
    }
]

results = list(collection.aggregate(pipeline))
program_counts = [doc['program_count'] for doc in results]

# Display statistics
print(f"\nTotal charities analyzed: {len(program_counts)}")
print(f"Charities with programs field: {sum(1 for x in program_counts if x > 0)}")
print(f"Charities without programs field: {sum(1 for x in program_counts if x == 0)}")
print(f"Minimum programs: {min(program_counts)}")
print(f"Maximum programs: {max(program_counts)}")
print(f"Average programs: {np.mean(program_counts):.2f}")
print(f"Median programs: {np.median(program_counts):.0f}")

# Create a figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

program_counts_filtered = [x for x in program_counts if x > 0]

# Plot 1: Programs 1-10
program_counts_1_to_10 = [x for x in program_counts if 1 <= x <= 10]
axes[0].hist(program_counts_1_to_10, bins=range(1, 12), 
             edgecolor='black', alpha=0.7, color='#5e3170', linewidth=1.5)
axes[0].set_xlabel('Number of Programs', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of Charities', fontsize=13, fontweight='bold')
axes[0].set_title('Charities with 1-10 Programs', fontsize=15, fontweight='bold', pad=15)
axes[0].set_xticks(range(1, 11))
axes[0].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
axes[0].tick_params(axis='both', labelsize=11)

# Add value labels on top of each bar
for i in range(1, 11):
    count = len([x for x in program_counts if x == i])
    if count > 0:
        axes[0].text(i, count, f'{count:,}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')

# Plot 2: Programs 11+
program_counts_11_plus = [x for x in program_counts if x > 10]
if program_counts_11_plus:
    bins_11_plus = sorted(set(program_counts_11_plus))
    axes[1].hist(program_counts_11_plus, bins=range(11, max(program_counts_filtered) + 2), 
                 edgecolor='black', alpha=0.7, color='#d62728', linewidth=1.5)
    axes[1].set_xlabel('Number of Programs', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Number of Charities', fontsize=13, fontweight='bold')
    axes[1].set_title(f'Charities with 11+ Programs (Total: {charities_with_10plus_programs})', 
                     fontsize=15, fontweight='bold', pad=15)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    axes[1].set_xlim(10.5, max(program_counts_filtered) + 0.5)
    axes[1].tick_params(axis='both', labelsize=11)
    
    # Add value labels for each bar
    for val in bins_11_plus:
        count = len([x for x in program_counts if x == val])
        if count > 0:
            axes[1].text(val, count, str(count), ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('programs_histogram_final.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved histogram as 'programs_histogram_final.png'")
plt.show()

# Optional: Show distribution table
print("\n--- Program Count Distribution ---")
unique_counts = sorted(set(program_counts))
for count in unique_counts:
    num_charities = program_counts.count(count)
    print(f"{count} programs: {num_charities} charities")

# Optional: List charities with more than 10 programs
print("\n--- Charities with more than 10 programs ---")
charities_detailed = collection.find(
    {
        "programs": {"$exists": True},
        "$expr": {"$gt": [{"$size": "$programs"}, 10]}
    },
    {"selected_name_final_normalized": 1, "programs": 1}
)

for charity in charities_detailed:
    print(f"{charity.get('selected_name_final_normalized', 'Unknown')}: {len(charity.get('programs', []))} programs")

# Close connection
client.close()