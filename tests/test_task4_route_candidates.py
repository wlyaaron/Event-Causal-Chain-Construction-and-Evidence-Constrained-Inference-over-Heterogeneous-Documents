"""Small invariant tests for the gold-blind route and candidate functions."""
import json
import unittest

from experiments.task4_route_candidates import candidates, parse_route


class RouteCandidateTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"question": "D002 如何影响 D004？",
                        "documents": [{"doc_id": "D001"}, {"doc_id": "D002"},
                                      {"doc_id": "D003"}, {"doc_id": "D004"}],
                        "events": None, "causal_relations": None}
        self.route = {"route": "answer", "evidence_mode": "path",
                      "basis_ids": ["D002", "D004"]}

    def test_strict_refusal_has_no_candidates(self):
        route = {**self.route, "route": "strict_refusal", "evidence_mode": "none"}
        self.assertEqual(candidates(self.payload, {"evidence_chain": ["D002"]}, route), [])

    def test_original_and_shorter_sequences_are_retained(self):
        result = candidates(self.payload,
                            {"evidence_chain": ["D001", "D002", "D003", "D004"]},
                            self.route)
        chains = [item["chain"] for item in result]
        self.assertEqual(chains[0], ["D001", "D002", "D003", "D004"])
        self.assertIn(["D002", "D003", "D004"], chains)
        self.assertIn(["D002", "D004"], chains)

    def test_graph_edges_only_when_visible(self):
        no_graph = candidates(self.payload, None, self.route)
        self.payload["causal_relations"] = [{"cause_event_id": "D002",
                                              "result_event_id": "D004"}]
        with_graph = candidates(self.payload, None, self.route)
        self.assertNotIn(["D002", "D004"], [x["chain"] for x in no_graph])
        self.assertIn(["D002", "D004"], [x["chain"] for x in with_graph])

    def test_route_validation_rejects_illegal_basis(self):
        raw = json.dumps({"route": "answer", "evidence_mode": "anchor",
                          "requested_target": "事件", "basis_ids": ["D999"], "reason": "x"})
        with self.assertRaises(ValueError):
            parse_route(raw, {"D001"})


if __name__ == "__main__":
    unittest.main()
