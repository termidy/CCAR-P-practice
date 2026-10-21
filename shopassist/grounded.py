from dotenv import load_dotenv
from anthropic import Anthropic
from retrieval import BM25, chunk_by_section

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"
index = BM25(chunk_by_section(open("policy.txt").read()))

SYSTEM = (
    "You are ShopAssist. Answer in one sentence, only from the "
    "search results. If they do not answer the question, say "
    "that the return policy does not cover it."
)


def as_search_result(chunk):
    return {
        "type": "search_result",
        "source": "policy.txt#" + chunk["id"],
        "title": chunk["id"],
        "content": [{"type": "text", "text": chunk["text"]}],
        "citations": {"enabled": True},
    }


def ask(question):
    content = []
    for chunk in index.search(question):
        content.append(as_search_result(chunk))
    content.append({"type": "text", "text": question})
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        output_config={"effort": "low"},
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    answer = ""
    sources = []
    for block in response.content:
        if block.type == "text":
            answer += block.text
            if block.citations:
                for citation in block.citations:
                    sources.append(citation.source)
    print("Q:", question)
    print("A:", answer)
    print("   cited:", sources)


ask("How long does a card refund take?")
ask("Do you ship to Canada?")
