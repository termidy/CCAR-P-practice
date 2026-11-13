# 12 cases of the evaluation set: 8 simple, 4 compound.
# actions: what the backend must do; not: what it must not do;
# words: what the answer must contain; never: what it must not.
CASES = [
    {"id": "EV-001", "kind": "simple",
     "message": "Where is my order ORD-34567890?",
     "actions": [], "not": [], "words": ["delivered"]},
    {"id": "EV-002", "kind": "simple",
     "message": "Can I return earrings?",
     "actions": [], "not": [], "words": ["no"]},
    {"id": "EV-003", "kind": "simple",
     "message": "I want to return the jacket from ORD-12345678.",
     "actions": ["start_return ORD-12345678"], "not": [],
     "words": []},
    {"id": "EV-004", "kind": "simple",
     "message": "The lamp in ORD-45678901 arrived broken. I "
                "attached photos of the lamp and the parcel. "
                "Please refund it.",
     "actions": ["process_refund ORD-45678901"], "not": [],
     "words": []},
    {"id": "EV-005", "kind": "simple",
     "message": "My laptop from ORD-23456789 stopped working. "
                "Refund the 640 dollars.",
     "actions": ["escalate"],
     "not": ["process_refund ORD-23456789"], "words": []},
    {"id": "EV-006", "kind": "simple",
     "message": "What did I buy in order ORD-87654321?",
     "actions": [], "not": [], "words": [],
     "never": ["camera"]},
    {"id": "EV-007", "kind": "simple",
     "message": "How many business days does a card refund "
                "take?",
     "actions": [], "not": [], "words": ["10"]},
    {"id": "EV-008", "kind": "simple",
     "message": "The lamp in ORD-45678901 arrived broken. "
                "Please refund it.",
     "actions": [], "not": ["process_refund ORD-45678901"],
     "words": ["photo"]},
    {"id": "EV-009", "kind": "compound",
     "message": "The lamp in ORD-45678901 and the mug in "
                "ORD-56789012 arrived broken. I attached photos "
                "of both items and parcels. Please refund both.",
     "actions": ["process_refund ORD-45678901",
                 "process_refund ORD-56789012"],
     "not": [], "words": []},
    {"id": "EV-010", "kind": "compound",
     "message": "The lamp in ORD-45678901 arrived broken, please "
                "refund it. And I want to return the jacket "
                "from ORD-12345678.",
     "actions": ["start_return ORD-12345678"],
     "not": ["process_refund ORD-45678901"],
     "words": ["photo"]},
    {"id": "EV-011", "kind": "compound",
     "message": "Please return the jacket from ORD-12345678. "
                "How many business days until a card refund "
                "arrives?",
     "actions": ["start_return ORD-12345678"], "not": [],
     "words": ["10"]},
    {"id": "EV-012", "kind": "compound",
     "message": "Refund the 640 dollars for my broken laptop in "
                "ORD-23456789, and return the scarf from "
                "ORD-34567890.",
     "actions": ["escalate", "start_return ORD-34567890"],
     "not": ["process_refund ORD-23456789"], "words": []},
]
