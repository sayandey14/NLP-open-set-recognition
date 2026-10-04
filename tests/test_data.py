from textbuckets.data import UNKNOWN, group_by_label, load_buckets, load_unknown_pool

EXPECTED_BUCKETS = {"Thoughts", "Feelings", "Journal", "Opinions", "Work", "Ideas", "To-dos"}
MIN_EXAMPLES_PER_BUCKET = 5


def test_buckets_have_expected_labels():
    assert set(group_by_label(load_buckets())) == EXPECTED_BUCKETS


def test_each_bucket_has_enough_examples():
    for label, texts in group_by_label(load_buckets()).items():
        assert len(texts) >= MIN_EXAMPLES_PER_BUCKET, label


def test_no_empty_or_duplicate_texts():
    texts = [item.text for item in load_buckets() + load_unknown_pool()]
    assert all(texts)
    assert len(texts) == len(set(texts))


def test_unknown_pool_is_disjoint_from_known_buckets():
    # If an unknown text's latent topic were a known bucket name, it wouldn't be "unknown".
    latent_topics = {item.label for item in load_unknown_pool()}
    assert latent_topics.isdisjoint(EXPECTED_BUCKETS | {UNKNOWN})


def test_group_by_label_preserves_order():
    groups = group_by_label(load_buckets())
    assert groups["To-dos"][0] == "Buy milk, eggs, and bread."
