import os
import pandas as pd
from openai import OpenAI
import time
from io import StringIO

# Initialize OpenAI client
client = OpenAI()

# Load your CSV file (ensure index is preserved)
print("Loading CSV file...")
df = pd.read_csv("/dccstor/gma2/mehuldamani/search-and-learn/cluster-questions/chat_questions_df.csv")
print(f"Loaded {len(df)} prompts.")

# System prompt with all three taxonomies
system_prompt = """You are an expert annotator for chatbot interactions.

For each chat question, perform classification along **three independent dimensions**:

### 1. Topic
- Medical Queries and Information
- Role-Playing Games
- Movie Reviews and Discussions
- SQL Database Table Queries
- Web Development Essentials
- Animal Behavior and Pet Care Queries
- Cooking and Recipes
- Email and Letter Writing Assistance
- Operations & Fleet Management
- Sports and Athletics Queries
- Advanced Mathematical Concepts
- Philosophical Texts & Concepts
- AI Impact and Applications
- Original Joke Requests
- Poetry Writing & Styles
- Word Play and Phonetics

### 2. Response Style
- Informational: factual queries, explanations, summaries
- Creative Generation: jokes, poems, role-play, stories
- Instructional: step-by-step "how-to" guidance
- Conversational / Opinionated: opinions, debates, casual chat
- Problem-Solving / Technical: coding, math, SQL, logic tasks

### 3. Cognitive Effort
- Low-effort retrieval: direct fact lookup
- Medium-effort synthesis: explanation, multi-part synthesis, comparisons
- High-effort reasoning/creativity: proofs, philosophical reasoning, complex problem-solving, creative writing

### Output Format
Return a valid CSV with five columns: 
- question_id
- question
- topic (from Topic list)
- response_style (from Response Style list)
- cognitive_effort (from Cognitive Effort list)

Make sure each row contains **all three labels** (topic, response style, cognitive effort) for the given question.
Do not output any other columns.
"""

# Parameters
batch_size = 25   # adjust based on token usage
results = []

total_batches = (len(df) + batch_size - 1) // batch_size
print(f"Processing in {total_batches} batches of up to {batch_size} prompts each.")

# Process in batches
for batch_num, i in enumerate(range(0, len(df), batch_size), 1):
    batch = df.iloc[i:i+batch_size]
    print(f"Processing batch {batch_num}/{total_batches} (rows {i} to {min(i+batch_size-1, len(df)-1)})...")

    # Build prompt text for this batch
    prompts_text = "Here are the chat questions:\n"
    for idx, row in batch.iterrows():
        prompts_text += f"{row['question_id']},{row['question']}\n"

    user_prompt = prompts_text + "\nClassify each row and output as CSV with columns: question_id,question,topic,response_style,cognitive_effort."

    try:
        print(f"Calling OpenAI API for batch {batch_num}...")
        response = client.chat.completions.create(
            model="gpt-5-mini",  # efficient for classification
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1
        )
        output = response.choices[0].message.content.strip()
        print(f"Received response for batch {batch_num}.")
        
        # Debug: Save raw GPT response
        # with open(f"debug_batch_{batch_num}_raw_response.txt", "w") as f:
        #     f.write(output)
        # print(f"Raw response saved to debug_batch_{batch_num}_raw_response.txt")

        # Parse the returned CSV text with robust parameters
        try:
            batch_results = pd.read_csv(StringIO(output), 
                                       quotechar='"', 
                                       skipinitialspace=True, 
                                       on_bad_lines='skip',
                                       engine='python')
            
            # Check if we got the expected columns
            expected_cols = ['question_id', 'question', 'topic', 'response_style', 'cognitive_effort']
            if not all(col in batch_results.columns for col in expected_cols):
                print(f"Warning: Batch {batch_num} missing expected columns. Got: {list(batch_results.columns)}")
                raise ValueError("Missing expected columns")
                
            print(f"Batch {batch_num} processed and parsed successfully.")
            print(f"Batch {batch_num} shape: {batch_results.shape}")
            print(f"Batch {batch_num} columns: {list(batch_results.columns)}")
            
        except Exception as parse_error:
            print(f"CSV parsing failed for batch {batch_num}: {parse_error}")
            print("Creating fallback batch with None values for classifications...")
            
            # Create fallback dataframe with original questions and None classifications
            batch_results = pd.DataFrame({
                'question_id': batch['question_id'].values,
                'question': batch['question'].values,
                'topic': None,
                'response_style': None,
                'cognitive_effort': None
            })
            print(f"Fallback batch {batch_num} created with {len(batch_results)} rows")

        results.append(batch_results)
        
        # save batch results
        # batch_results.to_csv(f"chat_questions_classified_{batch_num}.csv", index=False)

    except Exception as e:
        print(f"Error at batch {batch_num} (rows {i}-{min(i+batch_size-1, len(df)-1)}): {e}")
        time.sleep(5)

print("All batches processed. Concatenating results...")

# Merge all results
classified_df = pd.concat(results, ignore_index=True)

# Check classification success rate
successful_classifications = classified_df[classified_df['topic'].notna()]
failed_classifications = classified_df[classified_df['topic'].isna()]

print(f"Total rows processed: {len(classified_df)}")
print(f"Successfully classified: {len(successful_classifications)}")
print(f"Failed classifications (None values): {len(failed_classifications)}")
if len(failed_classifications) > 0:
    print("Failed question_ids:", failed_classifications['question_id'].tolist())

classified_df.set_index("question_id", inplace=True)

# Save raw results separately
raw_output_path = "/dccstor/gma2/mehuldamani/search-and-learn/cluster-questions/chat_questions_classified_raw_3.csv"
classified_df.to_csv(raw_output_path)
print(f"Raw results saved to {raw_output_path}")

# Join back with original dataframe on index
df = df.join(classified_df, how="left")

# Save final merged results
final_output_path = "/dccstor/gma2/mehuldamani/search-and-learn/cluster-questions/chat_questions_classified_3.csv"
df.to_csv(final_output_path)
print(f"Final merged file saved to {final_output_path}")

