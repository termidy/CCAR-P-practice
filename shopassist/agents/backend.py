# Stubs of DemoShop's order backend for the agents lab.
LIMIT = 500   # the automatic refund limit, in dollars

# set by the login, never by what Claude writes in a tool call
SESSION = {"customer_id": "CUS-1001"}

ORDERS = {
    "ORD-12345678": {"customer": "CUS-1001", "item": "jacket",
                     "price": 120, "delivered_days_ago": 20},
    "ORD-23456789": {"customer": "CUS-1001", "item": "laptop",
                     "price": 640, "delivered_days_ago": 10},
    "ORD-34567890": {"customer": "CUS-1001", "item": "scarf",
                     "price": 35, "delivered_days_ago": 12},
    "ORD-45678901": {"customer": "CUS-1001", "item": "lamp",
                     "price": 60, "delivered_days_ago": 3},
    "ORD-56789012": {"customer": "CUS-1001", "item": "mug",
                     "price": 25, "delivered_days_ago": 2},
    "ORD-87654321": {"customer": "CUS-2002", "item": "camera",
                     "price": 300, "delivered_days_ago": 5},
}


def error(category, customer_message, developer_message):
    return {"isError": True, "errorCategory": category,
            "retryable": False,
            "customerMessage": customer_message,
            "developerMessage": developer_message}


def own_order(order_id):
    order = ORDERS.get(order_id)
    if order is None:
        return None
    if order["customer"] != SESSION["customer_id"]:
        return None
    return order


def lookup_order(order_id):
    order = own_order(order_id)
    if order is None:
        return {"found": False,
                "message": "No order with this ID on your "
                           "account."}
    return {"found": True, "order_id": order_id,
            "item": order["item"], "price": order["price"],
            "status": "delivered",
            "delivered_days_ago": order["delivered_days_ago"]}


def start_return(order_id):
    order = own_order(order_id)
    if order is None:
        return error("permission",
                     "This order is not on your account.",
                     order_id + " is not the customer's order")
    if order["delivered_days_ago"] > 30:
        return error("business",
                     "This order is past the 30-day return "
                     "window.", "delivered_days_ago > 30")
    return {"return_id": "RET-" + order_id[-4:],
            "order_id": order_id}


def process_refund(order_id, amount, reason):
    order = own_order(order_id)
    if order is None:
        return error("permission",
                     "This order is not on your account.",
                     order_id + " is not the customer's order")
    if amount > order["price"]:
        return error("validation",
                     "The amount is more than the order total.",
                     f"amount {amount} > price {order['price']}")
    if amount > LIMIT:
        return error("business",
                     "A person must approve this refund.",
                     f"amount {amount} > limit {LIMIT}: escalate")
    return {"refund_id": "REF-" + order_id[-4:],
            "order_id": order_id, "amount": amount}


def escalate(reason):
    return {"escalation_id": "ESC-507",
            "customerMessage": "A person will reply within one "
                               "business day."}


FUNCTIONS = {"lookup_order": lookup_order,
             "start_return": start_return,
             "process_refund": process_refund,
             "escalate": escalate}
