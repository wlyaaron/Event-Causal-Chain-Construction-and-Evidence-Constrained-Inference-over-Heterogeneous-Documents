"""Meaningful invariants for the gold-blind A-view graph candidate generator."""

import unittest

from experiments.task4_a_candidate_pool import (
    enumerate_graph_paths,
    rank_graph_paths,
    rank_graph_paths_targeted,
)


class ACandidatePoolTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            {"event_id": event_id, "doc_id": event_id, "trigger_word": event_id}
            for event_id in ("D001", "D002", "D003", "D009")
        ]
        self.edges = [
            {"cause_event_id": "D001", "result_event_id": "D003"},
            {"cause_event_id": "D003", "result_event_id": "D002"},
            {"cause_event_id": "D002", "result_event_id": "D009"},
            {"cause_event_id": "D003", "result_event_id": "D009"},
        ]

    def test_numeric_id_reversal_is_a_valid_directed_path(self):
        paths = enumerate_graph_paths(self.events, self.edges)
        self.assertIn(("D001", "D003", "D002", "D009"), paths)
        self.assertIn(("D003", "D002"), paths)
        self.assertNotIn(("D002", "D003"), paths)

    def test_direct_and_multihop_paths_coexist(self):
        paths = enumerate_graph_paths(self.events, self.edges)
        self.assertIn(("D003", "D009"), paths)
        self.assertIn(("D003", "D002", "D009"), paths)
        self.assertIn(("D009",), paths)

    def test_explicit_endpoint_does_not_delete_other_paths(self):
        paths = enumerate_graph_paths(self.events, self.edges)
        ranked = rank_graph_paths(
            "围绕最终结果 D009，追溯它是如何形成的？",
            self.events,
            {event["doc_id"]: event["event_id"] for event in self.events},
            paths,
        )
        self.assertEqual(set(ranked), set(paths))

    def test_id_touching_chinese_is_seen_as_an_endpoint(self):
        paths = enumerate_graph_paths(self.events, self.edges)
        ranked = rank_graph_paths(
            "围绕最终结果D009，追溯其形成过程",
            self.events,
            {event["doc_id"]: event["event_id"] for event in self.events},
            paths,
            limit=3,
        )
        self.assertTrue(any(path[-1] == "D009" for path in ranked))

    def test_cap_raises_instead_of_returning_a_partial_pool(self):
        with self.assertRaisesRegex(ValueError, "safety cap"):
            enumerate_graph_paths(self.events, self.edges, max_paths=2)

    def test_direct_effect_only_prioritizes_visible_direct_outgoing_edges(self):
        events = [
            {"event_id": event_id, "doc_id": event_id, "trigger_word": event_id}
            for event_id in ("D001", "D002", "D003", "D004")
        ]
        edges = [
            {"cause_event_id": "D001", "result_event_id": "D002",
             "causal_type": "直接因果"},
            {"cause_event_id": "D001", "result_event_id": "D003",
             "causal_type": "间接传导"},
            {"cause_event_id": "D004", "result_event_id": "D001",
             "causal_type": "直接因果"},
        ]
        paths = enumerate_graph_paths(events, edges)
        ranked = rank_graph_paths_targeted(
            "D001（事故）之后直接发生了什么？", "retrospective", events, edges,
            {event["doc_id"]: event["event_id"] for event in events}, paths,
        )
        self.assertEqual(ranked[0], ("D001", "D002"))
        self.assertEqual(set(ranked), set(paths))

    def test_fact_question_prioritizes_its_single_event(self):
        paths = enumerate_graph_paths(self.events, self.edges)
        ranked = rank_graph_paths_targeted(
            "请说明D003的基本事实与直接影响。", "retrospective",
            self.events, self.edges,
            {event["doc_id"]: event["event_id"] for event in self.events},
            paths, limit=1,
        )
        self.assertEqual(ranked, [("D003",)])

    def test_other_question_types_and_two_id_questions_keep_original_order(self):
        paths = enumerate_graph_paths(self.events, self.edges)
        documents = {event["doc_id"]: event["event_id"] for event in self.events}
        for question, question_type in (
            ("D003之后直接发生什么？", "prospective"),
            ("D003直接引发D009了吗？", "retrospective"),
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    rank_graph_paths_targeted(
                        question, question_type, self.events, self.edges,
                        documents, paths, limit=3,
                    ),
                    rank_graph_paths(question, self.events, documents, paths,
                                     limit=3),
                )

    def test_temporal_causal_pair_promotes_only_visible_directed_pair(self):
        paths = enumerate_graph_paths(self.events, self.edges)
        documents = {event["doc_id"]: event["event_id"] for event in self.events}
        ranked = rank_graph_paths_targeted(
            "D003在D002之前，二者有因果关系吗？", "retrospective",
            self.events, self.edges, documents, paths, limit=5,
        )
        self.assertEqual(ranked[0], ("D003", "D002"))
        absent = rank_graph_paths_targeted(
            "D002在D003之前，二者有因果关系吗？", "retrospective",
            self.events, self.edges, documents, paths, limit=5,
        )
        self.assertEqual(absent, rank_graph_paths(
            "D002在D003之前，二者有因果关系吗？", self.events,
            documents, paths, limit=5,
        ))

    def test_unspecified_direct_prioritizes_both_edge_directions(self):
        events = self.events
        edges = [
            {"cause_event_id": "D001", "result_event_id": "D003",
             "causal_type": "直接因果"},
            {"cause_event_id": "D003", "result_event_id": "D002",
             "causal_type": "直接因果"},
            {"cause_event_id": "D003", "result_event_id": "D009",
             "causal_type": "间接传导"},
        ]
        paths = enumerate_graph_paths(events, edges)
        ranked = rank_graph_paths_targeted(
            "D003与哪一事件存在直接因果？", "retrospective", events, edges,
            {event["doc_id"]: event["event_id"] for event in events}, paths,
            limit=5,
        )
        self.assertEqual(set(ranked[:2]), {("D001", "D003"), ("D003", "D002")})
        self.assertNotIn(("D003", "D009"), ranked[:2])


if __name__ == "__main__":
    unittest.main()
