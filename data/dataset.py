from datasets import load_dataset


def stream_fineweb_for_tokenizer(
    target_mb: float = 200,
    save_path: str = "tokenizer_train.txt",
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    subset: str = "sample-10BT",
) -> str:
    """
    Stream FineWeb-Edu from HuggingFace until *target_mb* of text is collected,
    then write it to *save_path* and return the combined string.

    Documents are joined with ``\\n\\n<|endoftext|>\\n\\n`` so the tokenizer
    can learn the end-of-text boundary.

    Args:
        target_mb:    How many megabytes of text to collect.
        save_path:    Where to write the raw training text.
        dataset_name: HuggingFace dataset identifier.
        subset:       Dataset configuration / split name.

    Returns:
        The full combined text string.
    """
    print(f"Streaming {dataset_name} ({subset}) until {target_mb} MB collected...")

    target_bytes = int(target_mb * 1024 * 1024)

    ds = load_dataset(
        dataset_name,
        name=subset,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    collected: list[str] = []
    total_bytes = 0
    doc_count   = 0

    for doc in ds:
        text = doc["text"]
        if not text.strip():
            continue

        collected.append(text)
        total_bytes += len(text.encode("utf-8"))
        doc_count   += 1

        if doc_count % 1000 == 0:
            mb_so_far = total_bytes / 1024 ** 2
            print(f"  {doc_count:6,} docs | {mb_so_far:6.1f} MB / {target_mb} MB", end="\r")

        if total_bytes >= target_bytes:
            break

    print(
        f"\nCollected {doc_count:,} documents | "
        f"{total_bytes / 1024**2:.1f} MB of text"
    )

    combined = "\n\n<|endoftext|>\n\n".join(collected)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"Saved → {save_path}")
    return combined
