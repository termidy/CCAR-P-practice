import json
import sys
from datetime import date
from dotenv import load_dotenv
from anthropic import Anthropic
from retrieval import BM25, chunk_by_section

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"
LIVE = "policy_live.txt"


def rebuild():
    text = open(LIVE).read()
    index = {"built_from": text.split("\n")[0],
             "built_on": str(date.today()),
             "chunks": chunk_by_section(text)}
    json.dump(index, open("index.json", "w"), indent=1)
    print("index rebuilt from", index["built_from"])


def ask(question):
    index = json.load(open("index.json"))
    found = BM25(index["chunks"]).search(question, k=2)
    excerpts = "\n\n".join(chunk["text"] for chunk in found)
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        output_config={"effort": "low"},
        system="You are ShopAssist. Answer in one sentence "
               "from the policy excerpts.",
        messages=[{"role": "user", "content": f"<policy>\n"
                   f"{excerpts}\n</policy>\n\n{question}"}],
    )
    for block in response.content:
        if block.type == "text":
            print("A:", block.text)
    print("   retrieved:", [chunk["id"] for chunk in found])
    print("   index:", index["built_from"], index["built_on"])


if sys.argv[1:] == ["rebuild"]:
    rebuild()
else:
    ask("Can I still return a jacket delivered 20 days ago?")
