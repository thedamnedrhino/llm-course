## Generating (Summaries) Using Model and Tokenizer
```
def batch_generator(data: list, batch_size: int):
    """
    Creates batches of size `batch_size` from a list.
    """
    s = 0
    e = s + batch_size
    while s < len(data):
        yield data[s:e]
        s = e
        e = min(s + batch_size, len(data))


def summarize_with_t5(
    model_checkpoint: str, articles: list, batch_size: int = 8
) -> list:
    """
    Compute summaries using a T5 model.
    This is similar to a `pipeline` for a T5 model but does tokenization manually.

    :param model_checkpoint: Name for a model checkpoint in Hugging Face, such as "t5-small" or "t5-base"
    :param articles: List of strings, where each string represents one article.
    :return: List of strings, where each string represents one article's generated summary
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = T5ForConditionalGeneration.from_pretrained(
        model_checkpoint, cache_dir=DA.paths.datasets
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, model_max_length=1024, cache_dir=DA.paths.datasets
    )

    def perform_inference(batch: list) -> list:
        inputs = tokenizer(
            batch, max_length=1024, return_tensors="pt", padding=True, truncation=True
        )

        summary_ids = model.generate(
            inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            num_beams=2,
            min_length=0,
            max_length=40,
        )

        return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

    res = []

    summary_articles = list(map(lambda article: "summarize: " + article, articles))
    for batch in batch_generator(summary_articles, batch_size=batch_size):
        res += perform_inference(batch)

        torch.cuda.empty_cache()
        gc.collect()

    # clean up
    del tokenizer
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return res
```
