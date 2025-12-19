import os
import pandas as pd
from openai import OpenAI
import time
from io import StringIO

# Initialize OpenAI client (make sure OPENAI_API_KEY is set in your environment)
client = OpenAI()

# Load your CSV file
print("Loading CSV file...")
df = pd.read_csv("/dccstor/gma2/mehuldamani/search-and-learn/cluster-questions/math_questions_with_id.csv", quotechar='"', skipinitialspace=True, on_bad_lines='skip')
print(f"Loaded {len(df)} questions.")

# Define taxonomy and system prompt
system_prompt = """You are an expert math competition coach. 

For each math question, classify into two categories:

### Subdomains
- Algebra: equations, inequalities, polynomials, rational expressions, functional equations, sequences/series
- Geometry: plane/solid geometry, coordinate geometry, trigonometry, area/perimeter/volume
- Number Theory: primes, divisibility, modular arithmetic, Diophantine equations, gcd/lcm
- Counting & Combinatorics: permutations, combinations, binomial coefficients, arrangements
- Probability & Statistics: probability, expected value, variance, counting with randomness
- Calculus / Analysis: derivatives, integrals, limits, infinite series, approximation
- Discrete Math / Logic: graph theory, set theory, logic puzzles, recursions
- Other: problems that do not clearly fit into the above

### Formats
- Word Problem: problem stated in narrative or story form
- Symbolic Manipulation: simplify, differentiate, integrate, compute an exact expression
- Proof-Style: open-ended "prove that", "show that"
- Equation Solving: explicitly solve an equation/inequality/Diophantine problem
- Counting / Enumeration: explicit "how many ways" or combinatorial counting
- Geometry Construction / Diagram: geometry reasoning about figures, triangles, circles, etc.
- Other / Puzzle: logic puzzle or unconventional style


### Output Format
Return a valid CSV with four columns, the original question_id and question as well as the classified subdomain and format: question_id,question,subdomain,format
Make sure to include the question_id from the input for proper matching of question to classification.
"""

# Parameters
batch_size = 20   # adjust depending on token usage
results = []

total_batches = (len(df) + batch_size - 1) // batch_size
print(f"Processing in {total_batches} batches of up to {batch_size} questions each.")

# Process in batches
for batch_num, i in enumerate(range(0, len(df), batch_size), 1):
    batch = df.iloc[i:i+batch_size]
    print(f"Processing batch {batch_num}/{total_batches} (rows {i} to {min(i+batch_size-1, len(df)-1)})...")

    # Build prompt text for this batch
    problems_text = "Here are the problems, separated by question_id's:\n"
    for idx, row in batch.iterrows():
        problems_text += f"{row['question_id']},{row['question']}\n"

    user_prompt = problems_text + "\nClassify each question and output as a CSV the original question_id and question alongside the added subdomain and format. The final columns of the CSV should be: question_id, question, subdomain, format."

    # Call OpenAI API
    try:
        print(f"Calling OpenAI API for batch {batch_num}...")
        response = client.chat.completions.create(
            model="gpt-5-mini",  # recommended for classification
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1
        )
        output = response.choices[0].message.content.strip()
        print(f"Received response for batch {batch_num}.")

        # Parse the returned CSV text
        batch_results = pd.read_csv(StringIO(output), quotechar='"', skipinitialspace=True, on_bad_lines='skip')
        results.append(batch_results)
        print(f"Batch {batch_num} processed and parsed successfully.")
        print(f"Batch df dimensions: {batch_results.shape}")
        # save batch results
        # batch_results.to_csv(f"math_questions_{batch_num}.csv", index=False)

    except Exception as e:
        print(f"Error at batch {batch_num} (rows {i}-{min(i+batch_size-1, len(df)-1)}): {e}")
        time.sleep(5)  # wait before retrying

print("All batches processed. Saving intermediate results...")

# Save intermediate results first (in case postprocessing fails)
if results:
    classified_df = pd.concat(results, ignore_index=True)
    intermediate_path = "math_questions_classified_raw.csv"
    classified_df.to_csv(intermediate_path, index=False)
    print(f"Intermediate results saved to {intermediate_path}")
    print(f"Total classified rows: {len(classified_df)}")
    
    # Try to merge with original dataframe
    try:
        print("Attempting to merge with original dataframe using question_id...")
        
        # Merge on question_id to ensure correct matching
        final_df = df.merge(classified_df[['question_id', 'subdomain', 'format']], 
                           on='question_id', 
                           how='left')
        
        # Check for missing classifications
        missing_classifications = final_df[final_df['subdomain'].isna()]
        if len(missing_classifications) > 0:
            print(f"Warning: {len(missing_classifications)} questions missing classifications:")
            print(missing_classifications[['question_id', 'question']].head())
        
        # Save final merged file
        output_path = "math_questions_classified_final.csv"
        final_df.to_csv(output_path, index=False)
        print(f"Final merged results saved to {output_path}")
        print(f"Successfully classified {len(final_df[final_df['subdomain'].notna()])} out of {len(final_df)} questions")
            
    except Exception as merge_error:
        print(f"Error during merge: {merge_error}")
        print("Classified results are still saved in intermediate file.")
        
else:
    print("No results to save - all batches failed.")

