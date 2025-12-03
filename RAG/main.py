"""
Main file to run RAG system for sysadmin assistant.
"""

import argparse
import os
from rag_system import SysAdminRAG


def main():
    parser = argparse.ArgumentParser(description='RAG system for sysadmin assistant')
    parser.add_argument(
        '--mode',
        choices=['build', 'chat', 'test'],
        default='chat',
        help='Operation mode: build - build index, chat - interactive chat, test - testing'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/parsed.jsonl',
        help='Path to JSONL file with data'
    )
    parser.add_argument(
        '--index',
        type=str,
        default='colbert_index',
        help='Path for saving/loading index'
    )
    parser.add_argument(
        '--indexes',
        type=str,
        help='List of index paths (comma-separated) for multiple ColBERT shards'
    )
    parser.add_argument(
        '--auto-discover',
        action='store_true',
        default='True',
        help='Automatically find all indexes in base directory (--index)'
    )
    parser.add_argument(
        '--ollama-url',
        type=str,
        default='http://localhost:11434',
        help='Ollama server URL'
    )
    parser.add_argument(
        '--ollama-model',
        type=str,
        default='gemma:2b',
        help='Ollama model name'
    )
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Rebuild index (for build mode)'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Question for testing (for test mode)'
    )
    parser.add_argument(
        '--use-compression',
        action='store_true',
        help='Use LangChain compressors for document filtering'
    )
    parser.add_argument(
        '--compression-threshold',
        type=float,
        default=0.3,
        help='Similarity threshold for compression (0.0-1.0, default 0.3)'
    )
    
    args = parser.parse_args()
    
    # Parse index list (for sharding)
    index_paths = None
    if args.indexes:
        index_paths = [p.strip() for p in args.indexes.split(',') if p.strip()]
    
    # Initialize RAG system
    rag = SysAdminRAG(
        index_path=args.index,
        index_paths=index_paths,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        use_document_compression=args.use_compression,
        compression_similarity_threshold=args.compression_threshold
    )
    
    if args.mode == 'build':
        print("=" * 60)
        print("Building ColBERT Index")
        print("=" * 60)
        
        if not os.path.exists(args.data):
            print(f"Error: file {args.data} not found!")
            return
        
        rag.build_index(args.data, force_rebuild=args.rebuild)
        print("\nIndex successfully built!")
        
    elif args.mode == 'chat':
        print("=" * 60)
        print("Sysadmin Assistant - Interactive Mode")
        print("=" * 60)
        print("Loading index...")
        
        try:
            # Automatically find indexes if not explicitly specified
            rag.load_index(auto_discover=args.auto_discover or not args.indexes)
        except FileNotFoundError:
            print(f"Index not found. First run: python main.py --mode build")
            print("Or use --auto-discover for automatic index discovery")
            return
        
        print("Index loaded. Ready to work!")
        print("Enter 'quit' or 'exit' to exit.\n")
        
        while True:
            try:
                question = input("Question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\nSearching for relevant information...")
                result = rag.ask(question, k=5)
                
                print("\n" + "=" * 60)
                print("ANSWER:")
                print("=" * 60)
                print(result['answer'])
                
                if result['sources']:
                    print("\n" + "=" * 60)
                    print("SOURCES:")
                    print("=" * 60)
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. {source['title']}")
                        if source['url']:
                            print(f"   URL: {source['url']}")
                        print(f"   Relevance: {source['relevance_score']:.4f}")
                        print()
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
    
    elif args.mode == 'test':
        print("=" * 60)
        print("Test Mode")
        print("=" * 60)
        
        try:
            # Automatically find indexes if not explicitly specified
            rag.load_index(auto_discover=args.auto_discover or not args.indexes)
        except FileNotFoundError:
            print(f"Index not found. First run: python main.py --mode build")
            print("Or use --auto-discover for automatic index discovery")
            return
        
        if args.query:
            question = args.query
        else:
            # Example questions for testing
            test_questions = [
                "How to configure Active Directory on Ubuntu?",
                "How to backup the system?",
                "How to install Bacula?",
            ]
            question = test_questions[0]
            print(f"Using test question: {question}\n")
        
        print(f"Question: {question}\n")
        result = rag.ask(question, k=5)
        
        print("=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(result['answer'])
        
        if result['sources']:
            print("\n" + "=" * 60)
            print("SOURCES:")
            print("=" * 60)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['title']}")
                if source['url']:
                    print(f"   URL: {source['url']}")
                print(f"   Relevance: {source['relevance_score']:.4f}")
                print()


if __name__ == "__main__":
    main()

