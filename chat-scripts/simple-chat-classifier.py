import pandas as pd
import argparse
import sys

def classify_question(q: str) -> str:
    if not isinstance(q, str):
        return "Other / Uncategorized"
    
    q_lower = q.lower()

    # Expanded keyword lists per category
    math_keywords = [
        "solve", "equation", "math", "algebra", "geometry", "calculus",
        "integral", "derivative", "limit", "probability", "statistics",
        "matrix", "linear algebra", "function of x", "polynomial", "fraction",
        "roots", "square root", "theorem", "proof"
    ]

    programming_keywords = [
        "code", "python", "java", "c++", "javascript", "sql", "function",
        "program", "script", "debug", "error", "bug", "compile", "variable",
        "loop", "if statement", "recursion", "algorithm", "complexity",
        "data structure", "class", "object", "regex", "print", "return"
    ]

    writing_keywords = [
        "essay", "story", "poem", "novel", "creative", "summarize", "rewrite",
        "rephrase", "explain in simple terms", "expand", "shorten", "draft",
        "paragraph", "article", "speech", "letter", "dialogue", "review",
        "argument", "persuasive", "outline", "plot"
    ]

    translation_keywords = [
        "translate", "translation", "english", "spanish", "chinese", "french",
        "german", "japanese", "korean", "hindi", "italian", "portuguese",
        "language", "linguistic", "phrase", "sentence", "dictionary",
        "how to say", "in [language]"
    ]

    knowledge_keywords = [
        "history", "geography", "capital", "country", "city", "who is",
        "who was", "when was", "where is", "what is", "facts", "explain",
        "science", "physics", "chemistry", "biology", "law", "economics",
        "philosophy", "politics", "government", "famous", "inventor",
        "discovery", "war", "battle", "king", "queen", "president"
    ]

    advice_keywords = [
        "advice", "should i", "recommend", "recommendation", "opinion",
        "suggest", "help me decide", "which one", "choose", "career",
        "life", "relationship", "study tips", "health", "diet", "exercise",
        "how do i", "best way", "better", "improve", "optimize", "plan"
    ]

    # Check categories in order
    if any(word in q_lower for word in math_keywords):
        return "Mathematics"
    elif any(word in q_lower for word in programming_keywords):
        return "Programming"
    elif any(word in q_lower for word in writing_keywords):
        return "Writing-Creative"
    elif any(word in q_lower for word in translation_keywords):
        return "Translation"
    elif any(word in q_lower for word in knowledge_keywords):
        return "General Knowledge"
    elif any(word in q_lower for word in advice_keywords):
        return "Advice-Opinion"
    else:
        return "Other"

def add_question_type_column(df_train: pd.DataFrame) -> pd.DataFrame:
    if "question" not in df_train.columns:
        raise ValueError("DataFrame must contain a 'question' column.")
    types = []
    total = len(df_train)
    for idx, q in enumerate(df_train["question"], 1):
        t = classify_question(q)
        types.append(t)
        print(f"Classified question {idx}/{total}: {t}")
    df_train["type"] = types
    return df_train

def main():
    parser = argparse.ArgumentParser(description="Classify question types in a dataset and save the result.")
    parser.add_argument("input_file", help="Path to the input CSV file containing a 'question' column.")
    parser.add_argument("output_file", help="Path to save the output CSV file with the new 'type' column.")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df_with_type = add_question_type_column(df)
    except Exception as e:
        print(f"Error processing DataFrame: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df_with_type.to_csv(args.output_file, index=False)
        print(f"Output written to {args.output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
