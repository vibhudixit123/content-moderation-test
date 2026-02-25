import json
import random
from pymongo import MongoClient
from openai import OpenAI
from langsmith import Client
from langsmith.wrappers import wrap_openai
from typing import Dict, List, Tuple
import time
import os
from datetime import datetime
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGO_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
TARGET_DB = os.getenv('TARGET_DB')
TARGET_COLLECTION = os.getenv('TARGET_COLLECTION')

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME')
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.9'))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '2500'))

# LangSmith Configuration
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
LANGSMITH_PROJECT_PREFIX = os.getenv('LANGSMITH_PROJECT_PREFIX')
LANGSMITH_PROJECT = f'{LANGSMITH_PROJECT_PREFIX}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

# Processing Configuration
STOP_AFTER_N = int(os.getenv('STOP_AFTER_N')) if os.getenv('STOP_AFTER_N') else None
PROGRESS_FILE = os.getenv('PROGRESS_FILE')
GENERATED_SENTENCES_FILE = os.getenv('GENERATED_SENTENCES_FILE')


CATEGORY_DESCRIPTIONS = {
    "hate_speech_discrimination": {
        "description": "Content expressing hate, discrimination, or bias based on race, ethnicity, religion, or other protected characteristics",
        "themes": ["racial superiority", "ethnic discrimination", "religious intolerance", "xenophobia", "segregation"]
    },
    
    "politically_sensitive_partisan": {
        "description": "Content expressing explicit partisan political positions or endorsements inappropriate for charities",
        "themes": ["political endorsements", "partisan activities", "campaign support", "political ideology promotion", "election interference", "political leaders"]
    },
    
    "violence_extremism": {
        "description": "Content promoting violence, extremism, or militant action",
        "themes": ["armed resistance", "violent revolution", "militant tactics", "extremist ideologies", "inciting unrest"]
    },
    
    "provocative_inflammatory": {
        "description": "Content using provocative language, profanity, or inflammatory rhetoric",
        "themes": ["aggressive disruption", "profane demands", "hostile rhetoric", "confrontational stance", "destructive language"]
    },
    
    "harsh_unprofessional": {
        "description": "Content using harsh, insulting, or unprofessional language toward beneficiaries or stakeholders",
        "themes": ["insulting language", "dismissive attitude", "hostile demands", "unprofessional conduct", "abusive communication"]
    },
    
    "inappropriate_for_charities": {
        "description": "Content describing illegal, fraudulent, or unethical activities completely inappropriate for charitable organizations",
        "themes": ["illegal operations", "fraud", "exploitation", "criminal activities", "unethical practices"]
    }
}

CATEGORIES = list(CATEGORY_DESCRIPTIONS.keys())


def validate_env_variables():
    """Validate that all required environment variables are set"""
    required_vars = [
        'MONGO_CONNECTION_STRING',
        'TARGET_DB',
        'TARGET_COLLECTION',
        'OPENAI_API_KEY',
        'MODEL_NAME',
        'LANGSMITH_API_KEY',
        'LANGSMITH_PROJECT_PREFIX',
        'PROGRESS_FILE',
        'GENERATED_SENTENCES_FILE'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("ERROR: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease check your .env file")
        exit(1)
    
    print("✓ All required environment variables loaded")


def load_generated_sentences():
    """Load previously generated sentences from example.json"""
    try:
        if not os.path.exists(GENERATED_SENTENCES_FILE):
            # Create new file with empty structure
            initial_data = {category: [] for category in CATEGORIES}
            with open(GENERATED_SENTENCES_FILE, 'w') as f:
                json.dump(initial_data, f, indent=2)
            print(f"Created new {GENERATED_SENTENCES_FILE}")
            return initial_data
        
        with open(GENERATED_SENTENCES_FILE, 'r') as f:
            data = json.load(f)
        
        # Ensure all categories exist
        for category in CATEGORIES:
            if category not in data:
                data[category] = []
        
        print(f"\nLoaded generated sentences from {GENERATED_SENTENCES_FILE}:")
        for category, sentences in data.items():
            print(f"  - {category}: {len(sentences)} sentences")
        
        return data
        
    except Exception as e:
        print(f"ERROR loading {GENERATED_SENTENCES_FILE}: {e}")
        return {category: [] for category in CATEGORIES}


def save_generated_sentence(category: str, sentence: str):
    """Save a newly generated sentence to example.json"""
    try:
        data = load_generated_sentences()
        
        if sentence not in data[category]:
            data[category].append(sentence)
            
            with open(GENERATED_SENTENCES_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"  Saved sentence to {category} (total: {len(data[category])})")
        
    except Exception as e:
        print(f"  ERROR saving sentence: {e}")


def check_and_clear_old_run():
    """Check if this is a fresh run and clear old data if needed"""
    print("\nChecking for previous run data...")
    
    if not os.path.exists(PROGRESS_FILE):
        print("  No previous run found - starting fresh")
        return True
    
    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)
    
    current_prompt_hash = get_prompt_hash()
    old_prompt_hash = progress.get('prompt_hash', '')
    total_processed = progress.get('total_processed', 0)
    
    if current_prompt_hash != old_prompt_hash:
        print(f"  Prompt has changed!")
        print(f"    Old hash: {old_prompt_hash[:16]}...")
        print(f"    New hash: {current_prompt_hash[:16]}...")
        response = input("  Clear old data and start fresh? (yes/no): ")
        
        if response.lower() == 'yes':
            clear_all_data()
            return True
        else:
            print("  Continuing with old data (prompt mismatch may cause issues)")
            return False
    
    if total_processed > 0:
        print(f"  Found existing progress: {total_processed} documents processed")
        response = input("  Do you want to clear and start fresh? (yes/no): ")
        
        if response.lower() == 'yes':
            clear_all_data()
            return True
        else:
            print("  Continuing from existing progress")
            return False
    
    return False


def get_prompt_hash():
    """Generate hash of current prompts to detect changes"""
    system_prompt = get_system_prompt_sentence_generation()
    return hashlib.md5(system_prompt.encode()).hexdigest()


def clear_all_data():
    """Clear progress file and MongoDB AI_statement fields"""
    print("\nClearing all previous data...")
    
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("  Removed progress file")
    
    try:
        client = MongoClient(MONGO_CONNECTION_STRING)
        db = client[TARGET_DB]
        collection = db[TARGET_COLLECTION]
        
        result = collection.update_many(
            {'flagging': True},
            {'$set': {
                'AI_statement': '',
                'faulty_category': None,
                'faulty_content_inserted': None,
                'content_type': None,
                'insertion_location': None
            }}
        )
        
        print(f"  Cleared {result.modified_count} documents in MongoDB")
        client.close()
        
    except Exception as e:
        print(f"  ERROR clearing MongoDB: {e}")
    
    print("  Ready for fresh start\n")


def load_progress():
    """Load progress from file"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    
    return {
        'processed_ids': [],
        'category_counter': {cat: 0 for cat in CATEGORIES},
        'last_updated': None,
        'total_processed': 0,
        'prompt_hash': get_prompt_hash(),
        'sentence_count': 0
    }


def save_progress(progress):
    """Save progress to file"""
    progress['last_updated'] = datetime.now().isoformat()
    progress['prompt_hash'] = get_prompt_hash()
    
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def distribute_categories(total_statements: int, num_categories: int, existing_counter: Dict[str, int]) -> Dict[str, int]:
    """Distribute statements across categories evenly"""
    already_processed = sum(existing_counter.values())
    remaining = total_statements - already_processed
    
    if remaining <= 0:
        return existing_counter
    
    base_count = remaining // num_categories
    remainder = remaining % num_categories
    
    distribution = existing_counter.copy()
    for i, category in enumerate(CATEGORIES):
        distribution[category] += base_count + (1 if i < remainder else 0)
    
    return distribution


def create_statement_category_mapping(flagged_docs: List[Dict], distribution: Dict[str, int], processed_ids: List[str]) -> List[Tuple[Dict, str]]:
    """Create mapping of documents to categories"""
    unprocessed_docs = [doc for doc in flagged_docs if str(doc['_id']) not in processed_ids]
    
    mapping = []
    doc_index = 0
    
    for category in CATEGORIES:
        target_count = distribution[category]
        current_count = sum(1 for _, cat in mapping if cat == category)
        needed = target_count - current_count
        
        for _ in range(needed):
            if doc_index < len(unprocessed_docs):
                mapping.append((unprocessed_docs[doc_index], category))
                doc_index += 1
    
    random.shuffle(mapping)
    return mapping


def get_system_prompt_sentence_generation() -> str:
    """System prompt for GPT to generate varied sentences"""
    return """You are an expert at generating inappropriate content for charity mission statements for content moderation testing.

YOUR TASK: Generate a NEW, UNIQUE inappropriate sentence based on the category and themes provided.

CRITICAL RULES:
1. GENERATE a new sentence (1-2 lines) - DO NOT use templates or REPEAT PATTERNS
2. Make it specific to the category themes provided
3. The sentence should be clearly inappropriate but grammatically correct
4. Use varied language, different structures, diverse vocabulary
5. Ensure it feels like authentic inappropriate content, not formulaic
6. DO NOT replicate or closely paraphrase any of the previously generated sentences provided
7. Create genuinely different content - avoid similar patterns, structures, or wording

IMPORTANT:
- Create ORIGINAL content each time
- Use specific, concrete language
- Vary sentence structure and length
- Make it contextually inappropriate for a charity
- Keep it 1-2 lines maximum
- Ensure it's distinctly different from all previous examples"""


def get_user_prompt_sentence_generation(statement: str, category: str, description: str, themes: List[str], previous_sentences: List[str]) -> str:
    """User prompt for sentence generation with previous sentences context"""
    
    previous_examples = ""
    if previous_sentences:
        # Show up to last 20 sentences to avoid token limits
        recent_sentences = previous_sentences[-20:]
        previous_examples = f"""
PREVIOUSLY GENERATED SENTENCES FOR THIS CATEGORY:
(DO NOT replicate or closely paraphrase these - create something genuinely different)

{chr(10).join(f'{i+1}. "{sent}"' for i, sent in enumerate(recent_sentences))}

IMPORTANT: Your new sentence must be distinctly different from all of the above.
"""
    
    return f"""Original Charity Statement:
{statement}

Category: {category}
Description: {description}
Themes to incorporate: {', '.join(themes)}

{previous_examples}

TASK:
1. GENERATE a new inappropriate sentence (1-2 lines) based on the category and themes
2. Make it unique and specific - do not use generic templates
3. DO NOT replicate or closely paraphrase any of the previously generated sentences above
4. Insert it in the MIDDLE portion of the statement
5. Add proper punctuation to make it flow naturally
6. Keep ALL other text completely unchanged

REQUIREMENTS:
- Create ORIGINAL content (not templated)
- Use specific, concrete language related to the themes
- Make it clearly inappropriate for a charity
- Ensure grammatical correctness
- Keep it concise (1-2 lines maximum)
- Must be distinctly different from all previous examples

Return ONLY this JSON object:
{{
    "generated_sentence": "your newly generated inappropriate sentence here",
    "modified_statement": "the complete modified statement with your generated sentence inserted",
    "category_used": "{category}",
    "content_type": "generated_sentence",
    "insertion_location": "brief description of where inserted"
}}"""


def setup_langsmith_openai():
    """Setup OpenAI client with LangSmith wrapper"""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    langsmith_client = Client(api_key=LANGSMITH_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    wrapped_client = wrap_openai(openai_client)
    
    return wrapped_client, langsmith_client


def inject_faulty_content(client, statement: str, category: str, progress: Dict, generated_sentences: Dict) -> Dict:
    """Inject faulty content using sentence generation only"""
    
    description = CATEGORY_DESCRIPTIONS[category]['description']
    themes = CATEGORY_DESCRIPTIONS[category]['themes']
    previous_sentences = generated_sentences.get(category, [])
    
    system_prompt = get_system_prompt_sentence_generation()
    user_prompt = get_user_prompt_sentence_generation(
        statement, category, description, themes, previous_sentences
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        progress['sentence_count'] = progress.get('sentence_count', 0) + 1
        faulty_content = result.get('generated_sentence', 'N/A')
        
        # Save the generated sentence to example.json
        save_generated_sentence(category, faulty_content)
        
        return {
            'success': True,
            'modified_statement': result.get('modified_statement', ''),
            'category_used': result.get('category_used', category),
            'faulty_content_inserted': faulty_content,
            'content_type': 'generated_sentence',
            'insertion_location': result.get('insertion_location', 'middle'),
            'original_statement': statement
        }
    
    except Exception as e:
        print(f"  ERROR calling GPT: {e}")
        return {
            'success': False,
            'error': str(e),
            'original_statement': statement,
            'category_used': category
        }


def main():
    """Main execution function"""
    print("="*70)
    print("FAULTY CONTENT INJECTION - SENTENCE GENERATION ONLY")
    print("="*70)
    
    # Validate environment variables
    validate_env_variables()
    
    # Display configuration
    print("\nConfiguration from .env:")
    print(f"  Database: {TARGET_DB}.{TARGET_COLLECTION}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Max Tokens: {MAX_TOKENS}")
    print(f"  Stop After: {STOP_AFTER_N if STOP_AFTER_N else 'No limit'}")
    
    check_and_clear_old_run()
    
    print("\n1. Loading previously generated sentences from example.json...")
    generated_sentences = load_generated_sentences()
    
    print("\n2. Loading progress...")
    progress = load_progress()
    print(f"   Currently processed: {progress['total_processed']} documents")
    print(f"   - Generated sentences: {progress.get('sentence_count', 0)}")
    
    print("\n3. Connecting to MongoDB...")
    try:
        client = MongoClient(MONGO_CONNECTION_STRING)
        db = client[TARGET_DB]
        collection = db[TARGET_COLLECTION]
        print(f"   Connected to {TARGET_DB}.{TARGET_COLLECTION}")
    except Exception as e:
        print(f"   ERROR connecting to MongoDB: {e}")
        return
    
    print("\n4. Fetching documents...")
    flagged_docs = list(collection.find({'flagging': True}))
    total_flagged = len(flagged_docs)
    already_processed = len(progress['processed_ids'])
    remaining = total_flagged - already_processed
    
    docs_to_overwrite = collection.count_documents({
        'flagging': True,
        'AI_statement': {'$ne': ''},
        '_id': {'$nin': [doc['_id'] for doc in flagged_docs if str(doc['_id']) in progress['processed_ids']]}
    })
    
    print(f"   Total: {total_flagged}")
    print(f"   Processed: {already_processed}")
    print(f"   Remaining: {remaining}")
    if docs_to_overwrite > 0:
        print(f"   Will overwrite: {docs_to_overwrite} documents with existing AI_statement")
    
    if remaining == 0:
        print("\n   All documents already processed")
        client.close()
        return
    
    if STOP_AFTER_N:
        to_process = min(STOP_AFTER_N, remaining)
        print(f"\n   TEST MODE: Processing {to_process} documents")
    else:
        to_process = remaining
        print(f"\n   FULL MODE: Processing all {to_process} documents")
    
    print("\n5. Calculating distribution...")
    target_distribution = distribute_categories(total_flagged, len(CATEGORIES), progress['category_counter'])
    
    print("\n6. Creating mapping...")
    mapping = create_statement_category_mapping(flagged_docs, target_distribution, progress['processed_ids'])
    
    if STOP_AFTER_N and len(mapping) > STOP_AFTER_N:
        mapping = mapping[:STOP_AFTER_N]
    
    print(f"   Will process {len(mapping)} statements")
    
    print("\n7. Setting up GPT with LangSmith...")
    print(f"   Project: {LANGSMITH_PROJECT}")
    try:
        openai_client, langsmith_client = setup_langsmith_openai()
        print(f"   Ready to process")
    except Exception as e:
        print(f"   ERROR setting up OpenAI/LangSmith: {e}")
        client.close()
        return
    
    print("\n8. Processing statements...")
    print("="*70)
    
    success_counter = 0
    error_counter = 0
    
    for idx, (doc, category) in enumerate(mapping, 1):
        statement = doc.get('statementData', '')
        doc_id = doc['_id']
        existing_ai = doc.get('AI_statement', '')
        
        print(f"\n[{idx}/{len(mapping)}] Processing document")
        print(f"  ID: {doc_id}")
        print(f"  Category: {category}")
        print(f"  Previous sentences for {category}: {len(generated_sentences.get(category, []))}")
        
        if existing_ai:
            print(f"  NOTE: Overwriting existing AI_statement")
        
        result = inject_faulty_content(openai_client, statement, category, progress, generated_sentences)
        
        if result['success']:
            collection.update_one(
                {'_id': doc_id},
                {'$set': {
                    'AI_statement': result['modified_statement'],
                    'faulty_category': result['category_used'],
                    'faulty_content_inserted': result['faulty_content_inserted'],
                    'content_type': result['content_type'],
                    'insertion_location': result['insertion_location']
                }}
            )
            
            progress['processed_ids'].append(str(doc_id))
            progress['category_counter'][category] += 1
            progress['total_processed'] += 1
            save_progress(progress)
            
            # Reload generated sentences to get the updated list
            generated_sentences = load_generated_sentences()
            
            success_counter += 1
            
            print(f"  SUCCESS: Generated sentence")
            print(f"  Inserted: {result['faulty_content_inserted'][:80]}...")
            print(f"  Progress saved")
        else:
            error_counter += 1
            print(f"  ERROR: {result.get('error')}")
        
        if idx % 5 == 0:
            time.sleep(1)
    
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    print(f"Processed: {len(mapping)}")
    print(f"  Success: {success_counter}")
    print(f"  Errors: {error_counter}")
    print(f"\nTotal generated sentences: {progress.get('sentence_count', 0)}")
    print(f"\nSentences per category:")
    for cat in CATEGORIES:
        print(f"  {cat}: {len(generated_sentences.get(cat, []))}")
    print(f"\nTotal progress: {progress['total_processed']}/{total_flagged}")
    print(f"\nLangSmith Project: {LANGSMITH_PROJECT}")
    print(f"View at: https://smith.langchain.com/")
    print(f"\nGenerated sentences stored in: {GENERATED_SENTENCES_FILE}")
    
    if STOP_AFTER_N:
        print(f"\nTEST MODE: Set STOP_AFTER_N to empty in .env for full processing")
    
    print("="*70)
    
    client.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'clear':
            validate_env_variables()
            clear_all_data()
        elif sys.argv[1] == 'status':
            validate_env_variables()
            progress = load_progress()
            generated_sentences = load_generated_sentences()
            print("\nCurrent Progress:")
            print(f"  Total processed: {progress['total_processed']}")
            print(f"  Generated sentences: {progress.get('sentence_count', 0)}")
            print(f"  Last updated: {progress.get('last_updated')}")
            print(f"\nCategory Distribution:")
            for cat, count in progress['category_counter'].items():
                print(f"  {cat}: {count}")
            print(f"\nGenerated Sentences per Category:")
            for cat in CATEGORIES:
                print(f"  {cat}: {len(generated_sentences.get(cat, []))}")
    else:
        main()