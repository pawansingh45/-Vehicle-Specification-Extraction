import argparse
import json
import sys

from src.pipeline import SpecExtractionPipeline


DEMO_QUERIES = [
    "What is the torque specification for wheel nuts?",
    "What are the torque specs for suspension bolts?",
    "Stabilizer bar link torque specification",
    "Shock absorber mounting bolt torque",
    "Ball joint specifications",
    "Upper control arm bolt torque",
]


def main():
    parser = argparse.ArgumentParser(
        description="Vehicle Specification Extraction - RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --index                           Index the PDF
  python main.py --query "wheel nut torque"         Query for specs
  python main.py --batch                           Run demo queries
  python main.py --index --query "brake torque"    Index then query
  python main.py --reindex --batch --export json   Full pipeline
        """
    )

    parser.add_argument(
        "--index", action="store_true",
        help="Parse and index the PDF into the vector database"
    )
    parser.add_argument(
        "--reindex", action="store_true",
        help="Force re-indexing (clears existing data)"
    )
    parser.add_argument(
        "--query", type=str,
        help="Run a single query (e.g., 'Torque for brake caliper bolts')"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run a set of demo queries to showcase the pipeline"
    )
    parser.add_argument(
        "--export", type=str, choices=["json", "csv", "both"],
        help="Export results to file (json, csv, or both)"
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Number of chunks to retrieve per query (default: 8)"
    )

    args = parser.parse_args()

    if not any([args.index, args.reindex, args.query, args.batch]):
        parser.print_help()
        print("\n>> Quick start: python main.py --index --batch --export json")
        return

    pipeline = SpecExtractionPipeline()

    if args.index or args.reindex:
        pipeline.index_pdf(force_reindex=args.reindex)

    all_results = []

    if args.query:
        result = pipeline.query(args.query, top_k=args.top_k)
        all_results.append(result)
        _print_result(result)

    if args.batch:
        print("\n" + "=" * 60)
        print("  Running Demo Queries")
        print("=" * 60)
        results = pipeline.batch_query(DEMO_QUERIES)
        all_results.extend(results)
        for r in results:
            _print_result(r)

    if args.export and all_results:
        if args.export in ("json", "both"):
            pipeline.export_json(all_results)
        if args.export in ("csv", "both"):
            pipeline.export_csv(all_results)

    if all_results:
        total_specs = sum(len(r.get("specs", [])) for r in all_results)
        print(f"\n{'=' * 60}")
        print(f"  Summary: {len(all_results)} queries -> {total_specs} specs extracted")
        print(f"{'=' * 60}")


def _print_result(result: dict):
    print(f"\n{'=' * 60}")
    print(f"  Query: {result['query']}")
    print(f"{'=' * 60}")

    specs = result.get("specs", [])
    if not specs:
        print("  [!] No specifications found for this query.")
        return

    for i, spec in enumerate(specs, 1):
        print(f"\n  [{i}] {spec.get('component', 'N/A')}")
        print(f"      Type:    {spec.get('spec_type', 'N/A')}")
        print(f"      Value:   {spec.get('value', 'N/A')} {spec.get('unit', '')}")
        if spec.get('context'):
            ctx = spec['context'][:100]
            print(f"      Context: {ctx}")
        if spec.get('source_pages'):
            print(f"      Pages:   {spec['source_pages']}")

    print()


if __name__ == "__main__":
    main()
