COMMON_STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "can", "could", "would",
    "should", "will", "what", "why", "how", "when", "where", "which",
    "who", "and", "or", "but", "if", "matter", "explain", "describe",
    "like", "new", "programming", "actually", "someone", "beginner", "you",
}

def match_query(q_str, p_str):
    q_toks = [t.lower() for t in q_str.split() if t.isalnum()]
    p_toks = [t.lower() for t in p_str.split() if t.isalnum()]
    
    q_content = [t for t in q_toks if t not in COMMON_STOP_WORDS]
    p_content = [t for t in p_toks if t not in COMMON_STOP_WORDS]
    
    # Exact full match (covers "hello", "hi", "how are you")
    if q_toks == p_toks:
        return 100
        
    if not q_content or not p_content:
        # Fall back to token overlap for greetings
        overlap = len(set(q_toks) & set(p_toks))
        return overlap * 15 if overlap >= len(q_toks) // 2 else 0
        
    overlap = len(set(q_content) & set(p_content))
    if overlap == 0:
        return 0
    score = overlap * 30 - abs(len(q_content) - len(p_content)) * 5
    if set(q_content) == set(p_content):
        score += 40
    return score

print("Match 'does sun cause heat?' against 'why does dictionary matter':",
      match_query("does sun cause heat", "why does dictionary matter"))

print("Match 'what is default parameters?' against 'what is default parameters':",
      match_query("what is default parameters?", "what is default parameters"))

print("Match 'how are you' against 'how are you':",
      match_query("how are you", "how are you"))

print("Match 'hello' against 'hello':",
      match_query("hello", "hello"))
