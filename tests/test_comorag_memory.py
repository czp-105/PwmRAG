import unittest

from rag.state import ComoRAGMemoryNode, ComoRAGMemoryPool


class MemoryPoolTest(unittest.TestCase):
    def test_temporary_memory_is_committed_per_question(self):
        pool = ComoRAGMemoryPool()
        node = ComoRAGMemoryNode(
            probe="Who is Augusta's father?",
            kind="veridical",
            contents=("Augusta was a daughter of William I.",),
            cue="Augusta's father was William I.",
            source_ids=("2459", "poison-1"),
            poisoned_source_ids=("poison-1",),
        )

        pool.stage(node)
        self.assertEqual(pool.by_kind("veridical", temporary=True), [node])
        self.assertEqual(pool.by_kind("veridical"), [])
        self.assertEqual(pool.probes(), [])

        pool.commit()
        self.assertEqual(pool.temporary, [])
        self.assertEqual(pool.by_kind("veridical"), [node])
        self.assertEqual(pool.probes(), ["Who is Augusta's father?"])
        self.assertEqual(pool.main[0].poisoned_source_ids, ("poison-1",))
        self.assertEqual(ComoRAGMemoryPool().main, [])


if __name__ == "__main__":
    unittest.main()
