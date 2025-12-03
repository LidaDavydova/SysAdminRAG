"""
Утилита для разбиения большого parsed.jsonl на несколько файлов поменьше.

Пример:
    python split_parsed.py --input parsed.jsonl --out-prefix parsed_part --docs-per-file 1000
"""

import argparse
import os


def split_jsonl(input_path: str, out_prefix: str, docs_per_file: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(input_path)), exist_ok=True)

    part_idx = 1
    docs_in_current = 0
    out_f = None

    with open(input_path, "r", encoding="utf-8") as inp:
        for line in inp:
            if not line.strip():
                continue

            if out_f is None or docs_in_current >= docs_per_file:
                if out_f is not None:
                    out_f.close()
                out_name = f"{out_prefix}{part_idx}.jsonl"
                print(f"[INFO] Открываем файл {out_name}")
                out_f = open(out_name, "w", encoding="utf-8")
                docs_in_current = 0
                part_idx += 1

            out_f.write(line.rstrip("\n") + "\n")
            docs_in_current += 1

    if out_f is not None:
        out_f.close()
        print(f"[OK] Разбиение завершено, создано {part_idx - 1} файлов.")
    else:
        print("[WARN] Входной файл пустой, ничего не сделано.")


def main():
    parser = argparse.ArgumentParser(description="Разбить parsed.jsonl на несколько файлов.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/parsed.jsonl",
        help="Входной JSONL файл",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="data/parsed_part",
        help="Префикс для выходных файлов (будет добавлен номер части и .jsonl)",
    )
    parser.add_argument(
        "--docs-per-file",
        type=int,
        default=10,
        help="Сколько документов класть в один файл",
    )

    args = parser.parse_args()
    split_jsonl(args.input, args.out_prefix, args.docs_per_file)


if __name__ == "__main__":
    main()


