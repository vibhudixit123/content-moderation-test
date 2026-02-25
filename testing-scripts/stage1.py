"""
Charity Data Processor - Single Script Version
Process charity data from MongoDB:
1. Merge mission and program statements into statementData
2. Add wordCount field
3. Select top 5000 charities by word count
4. Add flagging and AI_statement columns
"""

from pymongo import MongoClient, UpdateOne
from typing import Dict, Any, List

# ============================================================================
# CONFIGURATION
# ============================================================================

MONGO_CONNECTION_STRING = 'mongodb://localhost:27017/'

SOURCE_DB = '53k'
SOURCE_COLLECTION = '77k'

TARGET_DB = 'content-moderation'
TARGET_COLLECTION = '5k'

TOP_N_CHARITIES = 5000
FLAG_THRESHOLD = 2500  # Top 2500 will have flagging=False

BATCH_SIZE = 100

# ============================================================================
# FUNCTIONS
# ============================================================================

def merge_statements(document: Dict[str, Any]) -> str:
    """Merge mission statement and program descriptions into a single paragraph"""
    statements = []
    
    # Add mission statement
    mission = document.get('selected_statement', '').strip()
    if mission:
        statements.append(mission)
    
    # Add program descriptions
    programs = document.get('programs', [])
    if programs and isinstance(programs, list):
        for program in programs:
            
            
            # Add program description
            desc = program.get('program_description_latest', '').strip()
            if desc:
                statements.append(desc)
    
    # Join all statements into a single paragraph
    merged_statement = ' '.join(statements)
    return merged_statement


def count_words(text: str) -> int:
    """Count words in a text string"""
    if not text:
        return 0
    return len(text.split())


def step1_add_statement_data(source_collection):
    """Step 1: Add statementData and wordCount to all documents"""
    print("\n" + "="*60)
    print("STEP 1: Adding statementData and wordCount")
    print("="*60)
    
    # Get all documents
    documents = list(source_collection.find())
    total_docs = len(documents)
    print(f"Processing {total_docs} documents...")
    
    updates = []
    updated_count = 0
    batch_num = 0
    
    for idx, doc in enumerate(documents, 1):
        # Merge statements
        merged_statement = merge_statements(doc)
        
        # Count words
        word_count = count_words(merged_statement)
        
        # Prepare update
        update = UpdateOne(
            {'_id': doc['_id']},
            {'$set': {
                'statementData': merged_statement,
                'wordCount': word_count
            }}
        )
        updates.append(update)
        
        # Bulk update in batches
        if len(updates) >= BATCH_SIZE:
            result = source_collection.bulk_write(updates)
            updated_count += result.modified_count
            batch_num += 1
            print(f"  Batch {batch_num}: Updated {result.modified_count} documents ({idx}/{total_docs})")
            updates = []
    
    # Update remaining documents
    if updates:
        result = source_collection.bulk_write(updates)
        updated_count += result.modified_count
        batch_num += 1
        print(f"  Batch {batch_num}: Updated {result.modified_count} documents (Final)")
    
    print(f"\n✓ Successfully updated {updated_count} documents")
    return updated_count


def step2_select_top_charities(source_collection, target_collection, top_n=5000):
    """Step 2: Select top N charities by word count and copy to target collection"""
    print("\n" + "="*60)
    print(f"STEP 2: Selecting top {top_n} charities by word count")
    print("="*60)
    
    # Clear target collection
    print("Clearing target collection...")
    result = target_collection.delete_many({})
    print(f"  Deleted {result.deleted_count} existing documents")
    
    # Get top documents sorted by wordCount
    top_documents = list(
        source_collection.find()
        .sort('wordCount', -1)  # -1 for descending
        .limit(top_n)
    )
    
    print(f"Found {len(top_documents)} documents to copy")
    
    if top_documents:
        max_words = top_documents[0].get('wordCount', 0)
        min_words = top_documents[-1].get('wordCount', 0)
        print(f"  Word count range: {min_words} - {max_words}")
    
    # Insert into target collection
    if top_documents:
        result = target_collection.insert_many(top_documents)
        inserted_count = len(result.inserted_ids)
        print(f"\n✓ Successfully copied {inserted_count} documents to target collection")
        return inserted_count
    else:
        print("\n✗ No documents found to copy")
        return 0


def step3_add_flagging_columns(target_collection, flag_threshold=2500):
    """Step 3: Add flagging and AI_statement columns"""
    print("\n" + "="*60)
    print("STEP 3: Adding flagging and AI_statement columns")
    print("="*60)
    print(f"  Top {flag_threshold} charities: flagging=False, AI_statement=statementData")
    print(f"  Remaining charities: flagging=True, AI_statement=''")
    
    # Get all documents from target collection, sorted by wordCount
    all_docs = list(
        target_collection.find()
        .sort('wordCount', -1)
    )
    
    total_docs = len(all_docs)
    print(f"\nProcessing {total_docs} documents...")
    
    updates = []
    flagged_false = 0
    flagged_true = 0
    batch_num = 0
    
    for idx, doc in enumerate(all_docs, 1):
        statement_data = doc.get('statementData', '')
        
        if idx <= flag_threshold:
            # Top N: flagging=False, AI_statement has content
            update_data = {
                'flagging': False,
                'AI_statement': statement_data
            }
            flagged_false += 1
        else:
            # Remaining: flagging=True, AI_statement is empty
            update_data = {
                'flagging': True,
                'AI_statement': ''
            }
            flagged_true += 1
        
        update = UpdateOne(
            {'_id': doc['_id']},
            {'$set': update_data}
        )
        updates.append(update)
        
        # Bulk update in batches
        if len(updates) >= BATCH_SIZE:
            result = target_collection.bulk_write(updates)
            batch_num += 1
            print(f"  Batch {batch_num}: Updated {result.modified_count} documents ({idx}/{total_docs})")
            updates = []
    
    # Update remaining documents
    if updates:
        result = target_collection.bulk_write(updates)
        batch_num += 1
        print(f"  Batch {batch_num}: Updated {result.modified_count} documents (Final)")
    
    print(f"\n✓ Successfully added flagging columns:")
    print(f"  - {flagged_false} documents with flagging=False (have AI_statement)")
    print(f"  - {flagged_true} documents with flagging=True (empty AI_statement)")
    
    return flagged_false, flagged_true


def print_summary(source_collection, target_collection):
    """Print processing summary"""
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    
    source_count = source_collection.count_documents({})
    target_count = target_collection.count_documents({})
    processed_count = source_collection.count_documents({'statementData': {'$exists': True}})
    
    print(f"\nSource Collection ({SOURCE_DB}.{SOURCE_COLLECTION}):")
    print(f"  Total documents: {source_count}")
    print(f"  Processed (with statementData): {processed_count}")
    
    print(f"\nTarget Collection ({TARGET_DB}.{TARGET_COLLECTION}):")
    print(f"  Total documents: {target_count}")
    
    if target_count > 0:
        flagged_false = target_collection.count_documents({'flagging': False})
        flagged_true = target_collection.count_documents({'flagging': True})
        print(f"  Flagged as False (with AI_statement): {flagged_false}")
        print(f"  Flagged as True (empty AI_statement): {flagged_true}")
    
    print("="*60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*60)
    print("CHARITY DATA PROCESSOR")
    print("="*60)
    
    # Connect to MongoDB
    print(f"\nConnecting to MongoDB at {MONGO_CONNECTION_STRING}")
    client = MongoClient(MONGO_CONNECTION_STRING)
    
    # Test connection
    try:
        client.admin.command('ping')
        print("✓ Connected to MongoDB successfully")
    except Exception as e:
        print(f"✗ Failed to connect to MongoDB: {e}")
        return
    
    # Setup collections
    source_db = client[SOURCE_DB]
    source_collection = source_db[SOURCE_COLLECTION]
    
    target_db = client[TARGET_DB]
    target_collection = target_db[TARGET_COLLECTION]
    
    print(f"  Source: {SOURCE_DB}.{SOURCE_COLLECTION}")
    print(f"  Target: {TARGET_DB}.{TARGET_COLLECTION}")
    
    try:
        # Step 1: Add statementData and wordCount
        step1_add_statement_data(source_collection)
        
        # Step 2: Select top 5000 charities
        step2_select_top_charities(source_collection, target_collection, TOP_N_CHARITIES)
        
        # Step 3: Add flagging and AI_statement columns
        step3_add_flagging_columns(target_collection, FLAG_THRESHOLD)
        
        # Print summary
        print_summary(source_collection, target_collection)
        
        print("\n✓ Processing completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close connection
        client.close()
        print("\nMongoDB connection closed.")


if __name__ == "__main__":
    main()