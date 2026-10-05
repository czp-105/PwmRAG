import importlib
import threading
import unittest


class ParallelTest(unittest.TestCase):
    def test_ordered_parallel_map_runs_tasks_concurrently_and_keeps_input_order(self):
        module = importlib.import_module("run_rag_comorag")
        self.assertTrue(hasattr(module, "ordered_parallel_map"))
        barrier = threading.Barrier(2)

        def work(value):
            barrier.wait(timeout=1)
            return value * 10

        self.assertEqual(list(module.ordered_parallel_map(work, [2, 1], workers=2)), [20, 10])


if __name__ == "__main__":
    unittest.main()
