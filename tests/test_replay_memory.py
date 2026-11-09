import pandas as pd

from ml.continual_learning.replay_memory import ClassBalancedMemory


def make_df(classes, per_class=50):
    rows = [
        {"text": f"{c}-{i}", "label": str(c)}
        for c in classes
        for i in range(per_class)
    ]
    return pd.DataFrame(rows)


def test_capacity_never_exceeded_and_quota_shrinks():
    memory = ClassBalancedMemory(capacity=100, seed=0)

    memory.update(make_df(range(0, 5)), range(0, 5))
    assert len(memory.as_dataframe()) == 100
    assert all(len(df) == 20 for df in memory.slots.values())

    memory.update(make_df(range(5, 10)), range(5, 10))
    assert len(memory.as_dataframe()) == 100
    assert all(len(df) == 10 for df in memory.slots.values())


def test_old_exemplars_are_a_subset_of_earlier_ones():
    memory = ClassBalancedMemory(capacity=100, seed=0)
    memory.update(make_df(range(0, 5)), range(0, 5))
    before = set(memory.slots["0"]["text"])

    memory.update(make_df(range(5, 10)), range(5, 10))
    after = set(memory.slots["0"]["text"])

    assert after <= before and len(after) < len(before)


def test_small_class_is_not_padded():
    memory = ClassBalancedMemory(capacity=100, seed=0)
    memory.update(make_df([0], per_class=5), [0])
    assert len(memory.slots["0"]) == 5
